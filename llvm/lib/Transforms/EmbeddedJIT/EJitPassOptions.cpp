//===-- EJitPassOptions.cpp - Shared Command-Line Options -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
using namespace llvm;

cl::opt<bool> EJitNoGlobalCtors(
    "ejit-no-global-ctors", cl::init(false), cl::Hidden,
    cl::desc("Skip llvm.global_ctors; only generate static registry tables "
             "(for bare-metal / testing)"));
