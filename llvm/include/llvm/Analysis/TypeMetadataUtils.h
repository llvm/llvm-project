//===- TypeMetadataUtils.h - Utilities related to type metadata --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions that make it easier to manipulate type metadata
// for devirtualization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TYPEMETADATAUTILS_H
#define LLVM_ANALYSIS_TYPEMETADATAUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallSite.h"

namespace llvm {

class DominatorTree;

/// The type of CFI jumptable needed for a function.
enum CfiFunctionLinkage {
  CFL_Definition = 0,
  CFL_Declaration = 1,
  CFL_WeakDeclaration = 2
};

/// A call site that could be devirtualized.
struct DevirtCallSite {
  /// The offset from the address point to the virtual function.
  uint64_t Offset;
  /// The call site itself.
  CallSite CS;
};

/// Given a call to the intrinsic \@llvm.type.test, find all devirtualizable
/// call sites based on the call and return them in DevirtCalls.
void findDevirtualizableCallsForTypeTest(
    SmallVectorImpl<DevirtCallSite> &DevirtCalls,
    SmallVectorImpl<CallInst *> &Assumes, const CallInst *CI,
    DominatorTree &DT);

/// Given a call to the intrinsic \@llvm.type.checked.load, find all
/// devirtualizable call sites based on the call and return them in DevirtCalls.
void findDevirtualizableCallsForTypeCheckedLoad(
    SmallVectorImpl<DevirtCallSite> &DevirtCalls,
    SmallVectorImpl<Instruction *> &LoadedPtrs,
    SmallVectorImpl<Instruction *> &Preds, bool &HasNonCallUses,
    const CallInst *CI, DominatorTree &DT);
}

#endif
