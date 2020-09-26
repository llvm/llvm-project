//===-- P2.h - Top-level interface for P2 representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM P2 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2_H
#define LLVM_LIB_TARGET_P2_P2_H

#include "MCTargetDesc/P2MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
    class P2TargetMachine;
    class FunctionPass;

    FunctionPass *createP2ExpandPseudosPass(P2TargetMachine &tm);


} // end namespace llvm;

#endif