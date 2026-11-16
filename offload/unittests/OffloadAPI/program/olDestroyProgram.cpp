//===------- Offload API tests - olDestroyProgram -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyProgramTest = OffloadProgramTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyProgramTest);

TEST_P(olDestroyProgramTest, Success) {
  ASSERT_SUCCESS(olDestroyProgram(Program));
  Program = nullptr;
}

TEST_P(olDestroyProgramTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyProgram(nullptr));
}

using olDestroyProgramGlobalDtorTest = LaunchKernelTestBase;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyProgramGlobalDtorTest);

TEST_P(olDestroyProgramGlobalDtorTest, RunsGlobalDestructors) {
  SKIP_KNOWN_FAILURE(LevelZero{"unsupported feature"});

  RETURN_ON_FATAL_FAILURE(SetUpProgram("global_dtor"));

  ol_symbol_handle_t SetTarget;
  ASSERT_SUCCESS(olGetSymbol(Program, "set_dtor_target", OL_SYMBOL_KIND_KERNEL,
                             &SetTarget));

  // The destructor reports through memory that outlives the program.
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_MANAGED,
                            sizeof(uint32_t), &Mem));
  *static_cast<uint32_t *>(Mem) = 0;

  LaunchArgs.GroupSize = {1, 1, 1};
  void *ArgPtrs[] = {&Mem};
  size_t ArgSizes[] = {sizeof(Mem)};
  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, SetTarget, &LaunchArgs, nullptr,
                                std::size(ArgPtrs), ArgPtrs, ArgSizes));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_EQ(*static_cast<uint32_t *>(Mem), 0u);

  ASSERT_SUCCESS(olDestroyProgram(Program));
  Program = nullptr;

  ASSERT_EQ(*static_cast<uint32_t *>(Mem), 0xD7u);
  ASSERT_SUCCESS(olMemFree(Context, Mem));
}
