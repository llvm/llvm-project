#include "orc-rt/cpu_feature_detect/FeatureDetect.h"
#include <sys/sysctl.h>

namespace orc_rt {

std::vector<std::string_view> detectLLVMFeatures() {
  std::vector<std::string_view> Features;
  auto CheckFlag = [](const char *Flag) noexcept -> bool {
    int V = 0;
    size_t S = sizeof(V);
    return sysctlbyname(Flag, &V, &S, nullptr, 0) == 0 && V != 0;
  };

#if defined(__x86_64__)
  if (CheckFlag("hw.optional.sse4_1"))
    Features.push_back(feature_name::x86::sse4_1);
  if (CheckFlag("hw.optional.sse4_2"))
    Features.push_back(feature_name::x86::sse4_2);
  if (CheckFlag("hw.optional.avx1_0"))
    Features.push_back(feature_name::x86::avx);
  if (CheckFlag("hw.optional.avx2_0"))
    Features.push_back(feature_name::x86::avx2);

#elif defined(__arm64__) || defined(__aarch64__)
  if (CheckFlag("hw.optional.arm.FEAT_DotProd"))
    Features.push_back(feature_name::aarch64::dotprod);
  if (CheckFlag("hw.optional.arm.FEAT_FP16"))
    Features.push_back(feature_name::aarch64::fullfp16);
  if (CheckFlag("hw.optional.arm.FEAT_SHA3"))
    Features.push_back(feature_name::aarch64::sha3);

  Features.push_back(
      feature_name::aarch64::neon); // on darwin NEON is always supported
#endif

  return Features;
}

} // namespace orc_rt
