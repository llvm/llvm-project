//===-- EJitPassOptions.cpp - Shared Command-Line Options -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
using namespace llvm;

cl::opt<bool> EnableEJitGlobalCtors(
    "enable-ejit-global-ctors", cl::init(true), cl::Hidden,
    cl::desc("Generate llvm.global_ctors for auto-registration "
             "(disable for bare-metal / testing)"));
