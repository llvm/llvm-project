#include "Shared/PluginAPI.h"
#include "omptarget.h"
#include "gtest/gtest.h"

namespace {
TEST(NextgenPluginsTest, PluginInit) {
  EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_init_plugin());
}
} // namespace
