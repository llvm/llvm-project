//===-- M88kISelLowering.cpp - M88k DAG lowering implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the M88kTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "M88kISelLowering.h"
#include "M88kTargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "m88k-lower"

M88kTargetLowering::M88kTargetLowering(const TargetMachine &TM,
                                       const M88kSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  addRegisterClass(MVT::i32, &M88k::GPRRCRegClass);
  addRegisterClass(MVT::f32, &M88k::GPRRCRegClass);
  addRegisterClass(MVT::f64, &M88k::GPR64RCRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties(Subtarget.getRegisterInfo());

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(M88k::R31);

  // How we extend i1 boolean values.
  setBooleanContents(ZeroOrOneBooleanContent);

  setMinFunctionAlignment(Align(4));
  setPrefFunctionAlignment(Align(4));
}
