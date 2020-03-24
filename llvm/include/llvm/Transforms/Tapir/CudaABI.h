//===- CudaABI.h - Interface to the Kitsune CUDA back end ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune CUDA ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//
#ifndef CUDA_ABI_H_
#define CUDA_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {

class DataLayout;
class TargetMachine;

class CudaABI : public TapirTarget {
public:
  CudaABI(Module &M) : TapirTarget(M) {}
  ~CudaABI() {}
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final {}
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final;
  void processSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL) const
    override final;
};

class PTXLoop : public LoopOutlineProcessor {
private:
  static unsigned NextKernelID;
  unsigned MyKernelID;
  Module PTXM;
  TargetMachine *PTXTargetMachine;
  GlobalVariable *PTXGlobal;

  FunctionCallee GetThreadIdx = nullptr;
  FunctionCallee GetBlockIdx = nullptr;
  FunctionCallee GetBlockDim = nullptr;
  FunctionCallee KitsuneCUDAInit = nullptr;
  FunctionCallee KitsuneGPUInitKernel = nullptr;
  FunctionCallee KitsuneGPUInitField = nullptr;
  FunctionCallee KitsuneGPUSetRunSize = nullptr;
  FunctionCallee KitsuneGPURunKernel = nullptr;
  FunctionCallee KitsuneGPUFinish = nullptr;
public:
  PTXLoop(Module &M);

  void setupLoopOutlineArgs(
      Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
      ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
      const SmallVectorImpl<Value *> &LCInputs,
      const ValueSet &TLInputsFixed)
    override final;
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};
}

#endif
