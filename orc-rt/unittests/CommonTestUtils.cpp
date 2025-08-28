//===- CommonTestUtils.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common test utilities.
//
//===----------------------------------------------------------------------===//

#include "CommonTestUtils.h"

size_t OpCounter::DefaultConstructions = 0;
size_t OpCounter::CopyConstructions = 0;
size_t OpCounter::CopyAssignments = 0;
size_t OpCounter::MoveConstructions = 0;
size_t OpCounter::MoveAssignments = 0;
size_t OpCounter::Destructions = 0;
