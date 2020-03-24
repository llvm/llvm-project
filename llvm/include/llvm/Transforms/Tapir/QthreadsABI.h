//===- QthreadsABI.h - Interface to the Qthreads runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Qthreads back end to convert Tapir instructions to
// calls into the Qthreads runtime system.
//
//===----------------------------------------------------------------------===//
#ifndef QTHREADS_ABI_H_
#define QTHREADS_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class QthreadsABI : public TapirTarget {
  ValueToValueMapTy SyncRegionToSinc;

  Type *QthreadFTy = nullptr;

  // Opaque Qthreads RTS functions
  FunctionCallee QthreadNumWorkers = nullptr;
  FunctionCallee QthreadForkCopyargs = nullptr;
  FunctionCallee QthreadInitialize = nullptr;
  FunctionCallee QtSincCreate = nullptr;
  FunctionCallee QtSincExpect = nullptr;
  FunctionCallee QtSincSubmit = nullptr;
  FunctionCallee QtSincWait = nullptr;
  FunctionCallee QtSincDestroy = nullptr;

  // Accessors for opaque Qthreads RTS functions
  FunctionCallee get_qthread_num_workers();
  FunctionCallee get_qthread_fork_copyargs();
  FunctionCallee get_qthread_initialize();
  FunctionCallee get_qt_sinc_create();
  FunctionCallee get_qt_sinc_expect();
  FunctionCallee get_qt_sinc_submit();
  FunctionCallee get_qt_sinc_wait();
  FunctionCallee get_qt_sinc_destroy();

  Value *getOrCreateSinc(Value *SyncRegion, Function *F);
public:
  QthreadsABI(Module &M);
  ~QthreadsABI() { SyncRegionToSinc.clear(); }

  ArgStructMode getArgStructMode() const override final {
    return ArgStructMode::Static;
  }
  Type *getReturnType() const override final {
    return Type::getInt32Ty(M.getContext());
  }

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final {}
  void processSpawner(Function &F) override final {}
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;
};

}  // end of llvm namespace

#endif
