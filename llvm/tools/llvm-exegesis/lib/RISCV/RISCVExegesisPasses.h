//===- RISCVExegesisPasses.h - RISC-V specific Exegesis Passes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_EXEGESIS_LIB_RISCV_RISCVEXEGESISPASSES_H
#define LLVM_TOOLS_EXEGESIS_LIB_RISCV_RISCVEXEGESISPASSES_H
namespace llvm {
class FunctionPass;

namespace exegesis {
FunctionPass *createRISCVPreprocessingPass();
FunctionPass *createRISCVPostprocessingPass();
} // namespace exegesis
} // namespace llvm
#endif
