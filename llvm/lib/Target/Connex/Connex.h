//===-- Connex.h - Top-level interface for Connex representation *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEX_H
#define LLVM_LIB_TARGET_CONNEX_CONNEX_H

#include "MCTargetDesc/ConnexMCTargetDesc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

// We define reserved register(s) of Connex to use for:
//    - handling COPY instructions in WHERE blocks
//     (see ConnexTargetMachine.cpp and ConnexISelLowering.cpp), etc
#define CONNEX_RESERVED_REGISTER_01 Connex::Wh30
#define CONNEX_RESERVED_REGISTER_02 Connex::Wh31
#define CONNEX_RESERVED_REGISTER_03 Connex::Wh29

// This definition is used also in the OPINCAA library
#define COPY_REGISTER_IMPLEMENTED_WITH_ORV_H

namespace llvm {
class ConnexTargetMachine;

FunctionPass *createConnexISelDag(ConnexTargetMachine &TM);

// Inspired from lib/Target/BPF/BPF.h (from Oct 2025)
void initializeConnexDAGToDAGISelLegacyPass(PassRegistry &);
} // End namespace llvm

#endif
