//===-- ToyMCTargetDesc.cpp - Toy Target Descriptions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Toy specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/ToyTargetInfo.h"
using namespace llvm;

extern "C" void LLVMInitializeToyTargetMC() {
  // TODO: We need this stub function definition during target registration,
  // otherwise we get an undefined symbol error. Appropriately fill-up this stub
  // function later.
}
