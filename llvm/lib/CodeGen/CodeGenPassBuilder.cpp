//===--- CodeGenPassBuilder.cpp --------------------------------------- ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/CodeGenPassBuilder.h"

using namespace llvm;

namespace llvm {
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                \
  MachinePassKey PASS_NAME::Key;
#include "llvm/CodeGen/MachinePassRegistry.def"
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)              \
  MachinePassKey PASS_NAME::Key;
#define DUMMY_MACHINE_FUNCTION_ANALYSIS(NAME, PASS_NAME, CONSTRUCTOR)          \
  AnalysisKey PASS_NAME::Key;
#include "llvm/CodeGen/MachinePassRegistry.def"
} // namespace llvm
