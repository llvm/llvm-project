//===------- Offload API tests - olCreateProgram --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCreateProgramTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateProgramTest);

TEST_P(olCreateProgramTest, Success) {

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  ol_program_handle_t Program;
  ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                 DeviceBin->getBufferSize(), &Program));
  ASSERT_NE(Program, nullptr);

  ASSERT_SUCCESS(olDestroyProgram(Program));
}

TEST_P(olCreateProgramTest, NullDeviceHandle) {
  
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  ol_program_handle_t Program;
  ASSERT_ERROR(
    OL_ERRC_INVALID_NULL_HANDLE, 
    olCreateProgram(nullptr, DeviceBin->getBufferStart(), DeviceBin->getBufferSize(), &Program)
  );

}

TEST_P(olCreateProgramTest, NullProgData) {
  
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  ol_program_handle_t Program;
  ASSERT_ERROR(
    OL_ERRC_INVALID_NULL_POINTER, 
    olCreateProgram(Device, nullptr, DeviceBin->getBufferSize(), &Program)
  );

}

TEST_P(olCreateProgramTest, NullOutputProgram) {
  
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  ASSERT_ERROR(
    OL_ERRC_INVALID_NULL_POINTER, 
    olCreateProgram(Device, DeviceBin->getBufferStart(), DeviceBin->getBufferSize(), nullptr)
  );

}