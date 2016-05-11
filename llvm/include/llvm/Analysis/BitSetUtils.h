//===- BitSetUtils.h - Utilities related to pointer bitsets ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions that make it easier to manipulate bitsets for
// devirtualization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BITSETUTILS_H
#define LLVM_ANALYSIS_BITSETUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallSite.h"

namespace llvm {

/// A call site that could be devirtualized.
struct DevirtCallSite {
  /// The offset from the address point to the virtual function.
  uint64_t Offset;
  /// The call site itself.
  CallSite CS;
};

/// Given a call to the intrinsic @llvm.bitset.test, find all devirtualizable
/// call sites based on the call and return them in DevirtCalls.
void findDevirtualizableCalls(SmallVectorImpl<DevirtCallSite> &DevirtCalls,
                              SmallVectorImpl<CallInst *> &Assumes,
                              CallInst *CI);
}

#endif
