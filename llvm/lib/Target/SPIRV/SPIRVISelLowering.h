//===-- SPIRVISelLowering.h - SPIR-V DAG Lowering Interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that SPIR-V uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVISELLOWERING_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class SPIRVSubtarget;

class SPIRVTargetLowering : public TargetLowering {
public:
  explicit SPIRVTargetLowering(const TargetMachine &TM,
                               const SPIRVSubtarget &STI)
      : TargetLowering(TM) {}

  // Stop IRTranslator breaking up FMA instrs to preserve types information.
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT) const override {
    return true;
  }

  // This is to prevent sexts of non-i64 vector indices which are generated
  // within general IRTranslator hence type generation for it is omitted.
  MVT getVectorIdxTy(const DataLayout &DL) const override {
    return MVT::getIntegerVT(32);
  }
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;
  MVT getRegisterTypeForCallingConv(LLVMContext &Context, CallingConv::ID CC,
                                    EVT VT) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVISELLOWERING_H
