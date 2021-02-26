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
#include "llvm/Analysis/TargetLibraryInfo.h"

namespace llvm {

class CallInst;
class DataLayout;
class Instruction;
class IntrinsicInst;
class Value;
class OptimizationRemarkEmitter;
class OptimizationRemarkMissed;
class StoreInst;

// FIXME: Once we get to more remarks like this one, we need to re-evaluate how
// much of this logic should actually go into the remark emitter.
struct AutoInitRemark {
  OptimizationRemarkEmitter &ORE;
  StringRef RemarkPass;
  const DataLayout &DL;
  const TargetLibraryInfo &TLI;

  AutoInitRemark(OptimizationRemarkEmitter &ORE, StringRef RemarkPass,
                 const DataLayout &DL, const TargetLibraryInfo &TLI)
      : ORE(ORE), RemarkPass(RemarkPass), DL(DL), TLI(TLI) {}

  void inspectStore(StoreInst &SI);
  void inspectUnknown(Instruction &I);
  void inspectIntrinsicCall(IntrinsicInst &II);
  void inspectCall(CallInst &CI);

private:
  template <typename FTy>
  void inspectCallee(FTy F, bool KnownLibCall, OptimizationRemarkMissed &R);
  void inspectKnownLibCall(CallInst &CI, LibFunc LF,
                           OptimizationRemarkMissed &R);
  void inspectSizeOperand(Value *V, OptimizationRemarkMissed &R);
};

} // namespace llvm

#endif
