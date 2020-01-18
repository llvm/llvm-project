//===- Xtensa.h - Top-level interface for Xtensa representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Xtensa back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSA_H
#define LLVM_LIB_TARGET_XTENSA_XTENSA_H

#include "MCTargetDesc/XtensaMCTargetDesc.h"
#include "llvm/PassRegistry.h"

namespace llvm {
class XtensaTargetMachine;
class FunctionPass;

FunctionPass *createXtensaISelDag(XtensaTargetMachine &TM,
                                  CodeGenOpt::Level OptLevel);
FunctionPass *createXtensaSizeReductionPass();
} // namespace llvm
#endif /* LLVM_LIB_TARGET_XTENSA_XTENSA_H */
