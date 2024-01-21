//===------- unittests/Plugins/NextgenPluginsTest.cpp - Plugin tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Shared/PluginAPI.h"
#include "omptarget.h"
#include "gtest/gtest.h"

#include <unordered_set>

const int DEVICE_ID = 0;
std::unordered_set<int> setup_map;

int init_test_device(int ID) {
  if (setup_map.find(ID) != setup_map.end()) {
    return OFFLOAD_SUCCESS;
  }
  if (__tgt_rtl_init_plugin() == OFFLOAD_FAIL ||
      __tgt_rtl_init_device(ID) == OFFLOAD_FAIL) {
    return OFFLOAD_FAIL;
  }
  setup_map.insert(ID);
  return OFFLOAD_SUCCESS;
}

// Test plugin initialization
TEST(NextgenPluginsTest, PluginInit) {
  EXPECT_EQ(OFFLOAD_SUCCESS, init_test_device(DEVICE_ID));
}

// Test GPU allocation and R/W
TEST(NextgenPluginsTest, PluginAlloc) {
  int32_t test_value = 23;
  int32_t host_value = -1;
  int64_t var_size = sizeof(int32_t);

  // Init plugin and device
  EXPECT_EQ(OFFLOAD_SUCCESS, init_test_device(DEVICE_ID));

  // Allocate memory
  void *device_ptr =
      __tgt_rtl_data_alloc(DEVICE_ID, var_size, nullptr, TARGET_ALLOC_DEFAULT);

  // Check that the result is not null
  EXPECT_NE(device_ptr, nullptr);

  // Submit data to device
  EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_submit(DEVICE_ID, device_ptr,
                                                   &test_value, var_size));

  // Read data from device
  EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_retrieve(DEVICE_ID, &host_value,
                                                     device_ptr, var_size));

  // Compare values
  EXPECT_EQ(host_value, test_value);

  // Cleanup data
  EXPECT_EQ(OFFLOAD_SUCCESS,
            __tgt_rtl_data_delete(DEVICE_ID, device_ptr, TARGET_ALLOC_DEFAULT));
}

// Test async GPU allocation and R/W
TEST(NextgenPluginsTest, PluginAsyncAlloc) {
  int32_t test_value = 47;
  int32_t host_value = -1;
  int64_t var_size = sizeof(int32_t);
  __tgt_async_info *info;

  // Init plugin and device
  EXPECT_EQ(OFFLOAD_SUCCESS, init_test_device(DEVICE_ID));

  // Check if device supports async
  // Platforms like x86_64 don't support it
  if (__tgt_rtl_init_async_info(DEVICE_ID, &info) == OFFLOAD_SUCCESS) {
    // Allocate memory
    void *device_ptr = __tgt_rtl_data_alloc(DEVICE_ID, var_size, nullptr,
                                            TARGET_ALLOC_DEFAULT);

    // Check that the result is not null
    EXPECT_NE(device_ptr, nullptr);

    // Submit data to device asynchronously
    EXPECT_EQ(OFFLOAD_SUCCESS,
              __tgt_rtl_data_submit_async(DEVICE_ID, device_ptr, &test_value,
                                          var_size, info));

    // Wait for async request to process
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_synchronize(DEVICE_ID, info));

    // Read data from device
    EXPECT_EQ(OFFLOAD_SUCCESS,
              __tgt_rtl_data_retrieve_async(DEVICE_ID, &host_value, device_ptr,
                                            var_size, info));

    // Wait for async request to process
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_synchronize(DEVICE_ID, info));

    // Compare values
    EXPECT_EQ(host_value, test_value);

    // Cleanup data
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_delete(DEVICE_ID, device_ptr,
                                                     TARGET_ALLOC_DEFAULT));
  }
}

// Test GPU data exchange
TEST(NextgenPluginsTest, PluginDataSwap) {
  int32_t test_value = 23;
  int32_t host_value = -1;
  int64_t var_size = sizeof(int32_t);

  // Look for compatible device
  int DEVICE_TWO = -1;
  for (int i = 1; i < __tgt_rtl_number_of_devices(); i++) {
    if (__tgt_rtl_is_data_exchangable(DEVICE_ID, i)) {
      DEVICE_TWO = i;
      break;
    }
  }

  // Only run test if we have multiple GPUs to test
  // GPUs must be compatible for test to work
  if (DEVICE_TWO >= 1) {
    // Init both GPUs
    EXPECT_EQ(OFFLOAD_SUCCESS, init_test_device(DEVICE_ID));
    EXPECT_EQ(OFFLOAD_SUCCESS, init_test_device(DEVICE_TWO));

    // Allocate memory on both GPUs
    // DEVICE_ID will be the source
    // DEVICE_TWO will be the destination
    void *source_ptr = __tgt_rtl_data_alloc(DEVICE_ID, var_size, nullptr,
                                            TARGET_ALLOC_DEFAULT);
    void *dest_ptr = __tgt_rtl_data_alloc(DEVICE_TWO, var_size, nullptr,
                                          TARGET_ALLOC_DEFAULT);

    // Check for success in allocation
    EXPECT_NE(source_ptr, nullptr);
    EXPECT_NE(dest_ptr, nullptr);

    // Write data to source
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_submit(DEVICE_ID, source_ptr,
                                                     &test_value, var_size));

    // Transfer data between devices
    EXPECT_EQ(OFFLOAD_SUCCESS,
              __tgt_rtl_data_exchange(DEVICE_ID, source_ptr, DEVICE_TWO,
                                      dest_ptr, var_size));

    // Read from destination device (DEVICE_TWO) memory
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_retrieve(DEVICE_TWO, &host_value,
                                                       dest_ptr, var_size));

    // Ensure match
    EXPECT_EQ(host_value, test_value);

    // Cleanup
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_delete(DEVICE_ID, source_ptr,
                                                     TARGET_ALLOC_DEFAULT));
    EXPECT_EQ(OFFLOAD_SUCCESS, __tgt_rtl_data_delete(DEVICE_TWO, dest_ptr,
                                                     TARGET_ALLOC_DEFAULT));
  }
}
