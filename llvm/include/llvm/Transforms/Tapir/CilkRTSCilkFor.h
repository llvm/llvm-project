//===- CilkRTSCilkFor.h - Interface to __cilkrts_cilk_for ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a loop-outline processor to lower Tapir loops to a call
// to a Cilk runtime method, __cilkrts_cilk_for.
//
//===----------------------------------------------------------------------===//
#ifndef CILKRTS_CILK_FOR_H_
#define CILKRTS_CILK_FOR_H_

#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {
class Value;
class TapirLoopInfo;

extern cl::opt<bool> UseRuntimeCilkFor;

/// The RuntimeCilkFor loop-outline processor transforms an outlined Tapir loop
/// to be processed using a call to a runtime method __cilkrts_cilk_for_32 or
/// __cilkrts_cilk_for_64.
class RuntimeCilkFor : public LoopOutlineProcessor {
  Type *GrainsizeType = nullptr;
  FunctionCallee CilkRTSCilkFor32 = nullptr;
  FunctionCallee CilkRTSCilkFor64 = nullptr;

  FunctionCallee Get__cilkrts_cilk_for_32();
  FunctionCallee Get__cilkrts_cilk_for_64();

public:
  RuntimeCilkFor(Module &M) : LoopOutlineProcessor(M) {
    GrainsizeType = Type::getInt32Ty(M.getContext());
  }

  ArgStructMode getArgStructMode() const override final {
    // return ArgStructMode::Dynamic;
    return ArgStructMode::Static;
  }
  void setupLoopOutlineArgs(
      Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
      ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
      const SmallVectorImpl<Value *> &LCInputs,
      const ValueSet &TLInputsFixed)
    override final;
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};
} // end namespace llvm

#endif
