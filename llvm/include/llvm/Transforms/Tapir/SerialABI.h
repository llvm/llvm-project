//===- SerialABI.h - Replace Tapir with serial projection ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Serial back end, which is used to convert Tapir
// instructions into their serial projection.
//
//===----------------------------------------------------------------------===//
#ifndef SERIAL_ABI_H_
#define SERIAL_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class SerialABI : public TapirTarget {
public:
  SerialABI(Module &M) : TapirTarget(M) {}
  ~SerialABI() {}

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &inst) override final;

  bool shouldDoOutlining(const Function &F) const override final {
    return false;
  }
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final {}
  void postProcessHelper(Function &F) override final {}

  void processOutlinedTask(Function &F) override final {}
  void processSpawner(Function &F) override final {}
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final {}
};

}  // end of llvm namespace

#endif
