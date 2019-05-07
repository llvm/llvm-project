//===-- DPU.h - Top-level interface for DPU representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM DPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DPU_H
#define LLVM_DPU_H

namespace llvm {
class FunctionPass;
class DPUTargetMachine;

FunctionPass *createDPUMergeComboInstrPass(DPUTargetMachine &tm);
FunctionPass *createDPUResolveMacroInstrPass(DPUTargetMachine &tm);

} // namespace llvm

#endif // LLVM_DPU_H
