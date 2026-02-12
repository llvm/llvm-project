//===- StridedMemRef.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/MemRefUtils.h"

#include "gmock/gmock.h"

using namespace ::mlir;
using namespace ::testing;

#ifndef __has_feature
#define __has_feature(x) 0
#endif

// hwaddress_sanitizer needs to be turned off for this move-assignment test
#if __has_feature(hwaddress_sanitizer)
#define MAYBE_assignOverloadChaining DISABLED_assignOverloadChaining
#else
#define MAYBE_assignOverloadChaining assignOverloadChaining
#endif

TEST(OwningMemRef, MAYBE_assignOverloadChaining) {
  int64_t mem1Shape[] = {3};
  int64_t mem2Shape[] = {4};

  OwningMemRef<float, 1> mem1(mem1Shape);
  OwningMemRef<float, 1> mem2(mem2Shape);
  OwningMemRef<float, 1> &ref = (mem1 = std::move(mem2));

  EXPECT_EQ(&ref, &mem1);
}
