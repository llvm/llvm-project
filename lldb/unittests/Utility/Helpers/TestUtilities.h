//===- TestUtilities.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
