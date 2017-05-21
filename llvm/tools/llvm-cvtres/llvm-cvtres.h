//===- llvm-cvtres.h ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMCVTRES_LLVMCVTRES_H
#define LLVM_TOOLS_LLVMCVTRES_LLVMCVTRES_H

#include <system_error>

void error(std::error_code EC);

enum class machine { UNKNOWN = 0, ARM, X64, X86 };

#endif
