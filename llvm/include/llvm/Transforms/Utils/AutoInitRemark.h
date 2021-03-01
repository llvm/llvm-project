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

  /// Emit a remark using information from the store's destination, size, etc.
  void inspectStore(StoreInst &SI);
  /// Emit a generic auto-init remark.
  void inspectUnknown(Instruction &I);
  /// Emit a remark using information from known intrinsic calls.
  void inspectIntrinsicCall(IntrinsicInst &II);
  /// Emit a remark using information from known function calls.
  void inspectCall(CallInst &CI);

private:
  /// Add callee information to a remark: whether it's known, the function name,
  /// etc.
  template <typename FTy>
  void inspectCallee(FTy F, bool KnownLibCall, OptimizationRemarkMissed &R);
  /// Add operand information to a remark based on knowledge we have for known
  /// libcalls.
  void inspectKnownLibCall(CallInst &CI, LibFunc LF,
                           OptimizationRemarkMissed &R);
  /// Add the memory operation size to a remark.
  void inspectSizeOperand(Value *V, OptimizationRemarkMissed &R);

  struct VariableInfo {
    Optional<StringRef> Name;
    Optional<uint64_t> Size;
    bool isEmpty() const { return !Name && !Size; }
  };
  /// Gather more information about \p V as a variable. This can be debug info,
  /// information from the alloca, etc. Since \p V can represent more than a
  /// single variable, they will all be added to the remark.
  void inspectDst(Value *Dst, OptimizationRemarkMissed &R);
  void inspectVariable(const Value *V, SmallVectorImpl<VariableInfo> &Result);
};

} // namespace llvm

#endif
