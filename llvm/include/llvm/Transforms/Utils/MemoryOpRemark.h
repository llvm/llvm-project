//===- MemoryOpRemark.h - Memory operation remark analysis -*- C++ ------*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Provide more information about instructions that copy, move, or initialize
// memory, including those with a "auto-init" !annotation metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MEMORYOPREMARK_H
#define LLVM_TRANSFORMS_UTILS_MEMORYOPREMARK_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

namespace llvm {

class CallInst;
class DataLayout;
class DiagnosticInfoIROptimization;
class Instruction;
class IntrinsicInst;
class Value;
class OptimizationRemarkEmitter;
class OptimizationRemarkMissed;
class StoreInst;

// FIXME: Once we get to more remarks like this one, we need to re-evaluate how
// much of this logic should actually go into the remark emitter.
struct MemoryOpRemark {
  OptimizationRemarkEmitter &ORE;
  StringRef RemarkPass;
  const DataLayout &DL;
  const TargetLibraryInfo &TLI;

  MemoryOpRemark(OptimizationRemarkEmitter &ORE, StringRef RemarkPass,
                 const DataLayout &DL, const TargetLibraryInfo &TLI)
      : ORE(ORE), RemarkPass(RemarkPass), DL(DL), TLI(TLI) {}

  virtual ~MemoryOpRemark();

  /// \return true iff the instruction is understood by MemoryOpRemark.
  static bool canHandle(const Instruction *I, const TargetLibraryInfo &TLI);

  void visit(const Instruction *I);

protected:
  virtual std::string explainSource(StringRef Type);

  enum RemarkKind { RK_Store, RK_Unknown, RK_IntrinsicCall, RK_Call };
  virtual StringRef remarkName(RemarkKind RK);

private:
  /// Emit a remark using information from the store's destination, size, etc.
  void visitStore(const StoreInst &SI);
  /// Emit a generic auto-init remark.
  void visitUnknown(const Instruction &I);
  /// Emit a remark using information from known intrinsic calls.
  void visitIntrinsicCall(const IntrinsicInst &II);
  /// Emit a remark using information from known function calls.
  void visitCall(const CallInst &CI);

  /// Add callee information to a remark: whether it's known, the function name,
  /// etc.
  template <typename FTy>
  void visitCallee(FTy F, bool KnownLibCall, OptimizationRemarkMissed &R);
  /// Add operand information to a remark based on knowledge we have for known
  /// libcalls.
  void visitKnownLibCall(const CallInst &CI, LibFunc LF,
                         OptimizationRemarkMissed &R);
  /// Add the memory operation size to a remark.
  void visitSizeOperand(Value *V, OptimizationRemarkMissed &R);

  struct VariableInfo {
    Optional<StringRef> Name;
    Optional<uint64_t> Size;
    bool isEmpty() const { return !Name && !Size; }
  };
  /// Gather more information about \p V as a variable. This can be debug info,
  /// information from the alloca, etc. Since \p V can represent more than a
  /// single variable, they will all be added to the remark.
  void visitPtr(Value *V, bool IsSrc, OptimizationRemarkMissed &R);
  void visitVariable(const Value *V, SmallVectorImpl<VariableInfo> &Result);
};

/// Special case for -ftrivial-auto-var-init remarks.
struct AutoInitRemark : public MemoryOpRemark {
  AutoInitRemark(OptimizationRemarkEmitter &ORE, StringRef RemarkPass,
                 const DataLayout &DL, const TargetLibraryInfo &TLI)
      : MemoryOpRemark(ORE, RemarkPass, DL, TLI) {}

  /// \return true iff the instruction is understood by AutoInitRemark.
  static bool canHandle(const Instruction *I);

protected:
  virtual std::string explainSource(StringRef Type) override;
  virtual StringRef remarkName(RemarkKind RK) override;
};

} // namespace llvm

#endif
