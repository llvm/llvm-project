//===- TestUtilities.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_UTILITY_HELPERS_TESTUTILITIES_H
#define LLDB_UNITTESTS_UTILITY_HELPERS_TESTUTILITIES_H

#include "llvm/ADT/Twine.h"
#include <string>

namespace lldb_private {
std::string GetInputFilePath(const llvm::Twine &name);
}

#endif
