//===------- Offload API tests - olGetKernelProperties --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olGetKernelPropertiesTest : OffloadProgramTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::SetUpWith("multiargs"));
    ASSERT_SUCCESS(olGetKernel(Device, "multiargs", &Kernel));
  }

  ol_kernel_handle_t Kernel = nullptr;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelPropertiesTest);

using olGetKernelPropertiesGlobalTest = OffloadGlobalTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelPropertiesGlobalTest);

TEST_P(olGetKernelPropertiesTest, Success) {
  ol_kernel_properties_t Properties{};
  ASSERT_SUCCESS(olGetKernelProperties(Kernel, &Properties));

  ASSERT_EQ(Properties.Version, 1u);
  ASSERT_TRUE(Properties.ValidFields & OL_KERNEL_PROPERTY_FLAG_NAME);
  ASSERT_STREQ(Properties.Name, "multiargs");
  ASSERT_TRUE(Properties.ValidFields & OL_KERNEL_PROPERTY_FLAG_MAX_NUM_THREADS);
  ASSERT_TRUE(Properties.ValidFields &
              OL_KERNEL_PROPERTY_FLAG_STATIC_SHARED_MEMORY_SIZE);

  if (Properties.ValidFields & OL_KERNEL_PROPERTY_FLAG_NUM_ARGS)
    ASSERT_GE(Properties.NumArgs, 3u);

  if (Properties.ValidFields & OL_KERNEL_PROPERTY_FLAG_ARG_SIZES) {
    ASSERT_NE(Properties.ArgSizes, nullptr);
    ASSERT_EQ(Properties.NumArgSizes, Properties.NumArgs);
    for (uint32_t I = 0; I < Properties.NumArgSizes; ++I)
      ASSERT_GT(Properties.ArgSizes[I], 0u);
  }
}

TEST_P(olGetKernelPropertiesTest, InvalidNullKernel) {
  ol_kernel_properties_t Properties{};
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelProperties(nullptr, &Properties));
}

TEST_P(olGetKernelPropertiesTest, InvalidNullProperties) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernelProperties(Kernel, nullptr));
}

TEST_P(olGetKernelPropertiesGlobalTest, InvalidSymbolKind) {
  ol_kernel_properties_t Properties{};
  ASSERT_ERROR(OL_ERRC_SYMBOL_KIND, olGetKernelProperties(Global, &Properties));
}
