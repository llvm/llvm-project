//===- AutoInitRemark.h - Auto-init remark analysis -*- C++ -------------*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Provide more information about instructions with a "auto-init"
// !annotation metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_AUTOINITREMARK_H
#define LLVM_TRANSFORMS_UTILS_AUTOINITREMARK_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class StoreInst;
class Instruction;
class OptimizationRemarkEmitter;
class DataLayout;

// FIXME: Once we get to more remarks like this one, we need to re-evaluate how
// much of this logic should actually go into the remark emitter.
struct AutoInitRemark {
  OptimizationRemarkEmitter &ORE;
  StringRef RemarkPass;
  const DataLayout &DL;

  AutoInitRemark(OptimizationRemarkEmitter &ORE, StringRef RemarkPass,
                 const DataLayout &DL)
      : ORE(ORE), RemarkPass(RemarkPass), DL(DL) {}

  void inspectStore(StoreInst &SI);
  void inspectUnknown(Instruction &I);
};

} // namespace llvm

#endif
