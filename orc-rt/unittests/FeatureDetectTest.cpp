#include "orc-rt/cpu_feature_detect/FeatureDetect.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace {

TEST(CpuFeatureDetectTest, FormatEmptyFeatures) {
  std::vector<std::string_view> V;

  EXPECT_EQ(formatLLVMFeatures(V), "");
}

TEST(CpuFeatureDetectTest, FormatSingleFeature) {
  std::vector<std::string_view> V = {feature_name::x86::avx2};

  EXPECT_EQ(formatLLVMFeatures(V), "+avx2");
}

TEST(CpuFeatureDetectTest, FormatMultipleFeatures) {
  std::vector<std::string_view> V = {feature_name::aarch64::neon,
                                     feature_name::aarch64::dotprod,
                                     feature_name::aarch64::fullfp16};

  EXPECT_EQ(formatLLVMFeatures(V), "+neon,+dotprod,+fullfp16");
}

TEST(CpuFeatureDetectTest, DetectDoesNotCrash) {
  [[maybe_unused]] auto _ = detectLLVMFeatures();
  SUCCEED();
}

// this test should catch any issues that arise from out of order tests
// hopefully.
TEST(CpuFeatureDetectTest, CachedResultIsIdempotent) {
  const std::string &First = LLVMFeatures();
  const std::string &Second = LLVMFeatures();

  EXPECT_EQ(First, Second);
  // has to not only be equal but the same address.
  EXPECT_EQ(&First, &Second);
}

TEST(CpuFeatureDetectTest, CachedStringMatchesFormatted) {
  std::string Formatted = formatLLVMFeatures(detectLLVMFeatures());
  std::string_view Cached = LLVMFeatures();

  EXPECT_EQ(Formatted, Cached);
}

} // anonymous namespace
