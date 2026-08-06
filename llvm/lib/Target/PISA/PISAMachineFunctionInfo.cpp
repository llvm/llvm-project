//===-- PISAMachineFunctionInfo.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAMachineFunctionInfo.h"
#include "PISASubtarget.h"

using namespace llvm;

PISAMachineFunctionInfo::PISAMachineFunctionInfo(const Function &F,
                                                 const PISASubtarget *STI) {}

PISAMachineFunctionInfo::~PISAMachineFunctionInfo() = default;
