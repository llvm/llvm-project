#ifndef ORC_RT_FEATURE_DETECT_H
#define ORC_RT_FEATURE_DETECT_H

#include <string>
#include <string_view>
#include <vector>

namespace orc_rt {
namespace feature_name {
namespace x86 {
inline constexpr std::string_view sse4_1 = "sse4.1";
inline constexpr std::string_view sse4_2 = "sse4.2";
inline constexpr std::string_view avx = "avx";
inline constexpr std::string_view avx2 = "avx2";
} // namespace x86
namespace aarch64 {
inline constexpr std::string_view neon = "neon";
inline constexpr std::string_view dotprod = "dotprod";
inline constexpr std::string_view fullfp16 = "fullfp16";
inline constexpr std::string_view sha3 = "sha3";
} // namespace aarch64
} // namespace feature_name

std::vector<std::string_view> detectLLVMFeatures();

inline std::string
formatLLVMFeatures(const std::vector<std::string_view> &Features) {
  if (Features.empty())
    return "";

  std::string R;
  R += '+';
  R += Features[0];
  for (size_t I = 1; I < Features.size(); I++) {
    R += ",+";
    R += Features[I];
  }
  return R;
}

inline const std::string &LLVMFeatures() { // NOLINT
  static const auto Cache = []() {
    return formatLLVMFeatures(detectLLVMFeatures());
  }();
  return Cache;
}

} // namespace orc_rt

#endif // ORC_RT_FEATURE_DETECT_H
