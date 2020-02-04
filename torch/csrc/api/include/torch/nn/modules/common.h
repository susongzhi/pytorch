#pragma once

// yf225 TODO: finish this up

/// This macro enables a module with default arguments in its forward method
/// to be used in a Sequential module.
/// 
/// Example usage:
Let's say we have a module declared like this:
```
// yf225 TODO check this module can actually be saved in a Sequential! and repro the problem we are trying to solve!

struct M : public torch::nn::Cloneable<M> {
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }
};
```



 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(2)}, {2, torch::nn::AnyValue(3.0)})
/// ```
/// class TORCH_API EmbeddingBagImpl : public torch::nn::Cloneable<EmbeddingBagImpl> {
///  public:
///   ...
///   Tensor forward(const Tensor& input, const Tensor& offsets = {}, const Tensor& per_sample_weights = {});
///  protected:
///   FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})
/// };
/// ```
#define FORWARD_HAS_DEFAULT_ARGS(...) \
  template <typename ModuleType, typename... ArgumentTypes> \
  friend class torch::nn::AnyModuleHolder; \
  bool _forward_has_default_args() override { \
    return true; \
  } \
  unsigned int _forward_num_required_args() override { \
    std::vector<std::pair<unsigned int, AnyValue>> args_info = {__VA_ARGS__}; \
    return args_info[0].first; \
  } \
  std::vector<AnyValue> _forward_populate_default_args(std::vector<AnyValue>&& arguments) override { \
    std::vector<std::pair<unsigned int, AnyValue>> args_info = {__VA_ARGS__}; \
    unsigned int num_all_args = args_info[args_info.size() - 1].first + 1; \
    TORCH_INTERNAL_ASSERT(arguments.size() >= _forward_num_required_args() && arguments.size() <= num_all_args); \
    std::vector<AnyValue> ret; \
    ret.reserve(num_all_args); \
    for (size_t i = 0; i < arguments.size(); i++) { \
      ret.emplace_back(std::move(arguments[i])); \
    } \
    for (auto& arg_info : args_info) \
      if (arg_info.first > ret.size() - 1) ret.emplace_back(std::move(arg_info.second)); \
    return std::move(ret); \
  }
