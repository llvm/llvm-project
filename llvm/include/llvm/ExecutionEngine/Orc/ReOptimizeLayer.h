//===- ReOptimizeLayer.h - Re-optimization layer interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Re-optimization layer interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_ORC_REOPTIMIZELAYER_H
#define LLVM_EXECUTIONENGINE_ORC_REOPTIMIZELAYER_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {
namespace orc {

class ReOptimizeLayer : public IRLayer, public ResourceManager {
public:
  using ReOptMaterializationUnitID = uint64_t;

  /// AddProfilerFunc will be called when ReOptimizeLayer emits the first
  /// version of a materialization unit in order to inject profiling code and
  /// reoptimization request code.
  using AddProfilerFunc = unique_function<Error(
      ReOptimizeLayer &Parent, ReOptMaterializationUnitID MUID,
      unsigned CurVersion, ThreadSafeModule &TSM)>;

  /// ReOptimizeFunc will be called when ReOptimizeLayer reoptimization of a
  /// materialization unit was requested in order to reoptimize the IR module
  /// based on profile data. OldRT is the ResourceTracker that tracks the old
  /// function definitions. The OldRT must be kept alive until it can be
  /// guaranteed that every invocation of the old function definitions has been
  /// terminated.
  using ReOptimizeFunc = unique_function<Error(
      ReOptimizeLayer &Parent, ReOptMaterializationUnitID MUID,
      unsigned CurVersion, ResourceTrackerSP OldRT, ThreadSafeModule &TSM)>;

  ReOptimizeLayer(ExecutionSession &ES, DataLayout &DL, IRLayer &BaseLayer,
                  RedirectableSymbolManager &RM)
      : IRLayer(ES, BaseLayer.getManglingOptions()), ES(ES), Mangle(ES, DL),
        BaseLayer(BaseLayer), RSManager(RM), ReOptFunc(identity),
        ProfilerFunc(reoptimizeIfCallFrequent) {}

  void setReoptimizeFunc(ReOptimizeFunc ReOptFunc) {
    this->ReOptFunc = std::move(ReOptFunc);
  }

  void setAddProfilerFunc(AddProfilerFunc ProfilerFunc) {
    this->ProfilerFunc = std::move(ProfilerFunc);
  }

  /// Registers reoptimize runtime dispatch handlers to given PlatformJD. The
  /// reoptimization request will not be handled if dispatch handler is not
  /// registered by using this function.
  Error reigsterRuntimeFunctions(JITDylib &PlatformJD);

  /// Emits the given module. This should not be called by clients: it will be
  /// called by the JIT when a definition added via the add method is requested.
  void emit(std::unique_ptr<MaterializationResponsibility> R,
            ThreadSafeModule TSM) override;

  static const uint64_t CallCountThreshold = 10;

  /// Basic AddProfilerFunc that reoptimizes the function when the call count
  /// exceeds CallCountThreshold.
  static Error reoptimizeIfCallFrequent(ReOptimizeLayer &Parent,
                                        ReOptMaterializationUnitID MUID,
                                        unsigned CurVersion,
                                        ThreadSafeModule &TSM);

  static Error identity(ReOptimizeLayer &Parent,
                        ReOptMaterializationUnitID MUID, unsigned CurVersion,
                        ResourceTrackerSP OldRT, ThreadSafeModule &TSM) {
    return Error::success();
  }

  // Create IR reoptimize request fucntion call.
  static void createReoptimizeCall(Module &M, Instruction &IP,
                                   GlobalVariable *ArgBuffer);

  Error handleRemoveResources(JITDylib &JD, ResourceKey K) override;
  void handleTransferResources(JITDylib &JD, ResourceKey DstK,
                               ResourceKey SrcK) override;

private:
  class ReOptMaterializationUnitState {
  public:
    ReOptMaterializationUnitState() = default;
    ReOptMaterializationUnitState(ReOptMaterializationUnitID ID,
                                  ThreadSafeModule TSM)
        : ID(ID), TSM(std::move(TSM)) {}
    ReOptMaterializationUnitState(ReOptMaterializationUnitState &&Other)
        : ID(Other.ID), TSM(std::move(Other.TSM)), RT(std::move(Other.RT)),
          Reoptimizing(std::move(Other.Reoptimizing)),
          CurVersion(Other.CurVersion) {}

    ReOptMaterializationUnitID getID() { return ID; }

    const ThreadSafeModule &getThreadSafeModule() { return TSM; }

    ResourceTrackerSP getResourceTracker() {
      std::unique_lock<std::mutex> Lock(Mutex);
      return RT;
    }

    void setResourceTracker(ResourceTrackerSP RT) {
      std::unique_lock<std::mutex> Lock(Mutex);
      this->RT = RT;
    }

    uint32_t getCurVersion() {
      std::unique_lock<std::mutex> Lock(Mutex);
      return CurVersion;
    }

    bool tryStartReoptimize();
    void reoptimizeSucceeded();
    void reoptimizeFailed();

  private:
    std::mutex Mutex;
    ReOptMaterializationUnitID ID;
    ThreadSafeModule TSM;
    ResourceTrackerSP RT;
    bool Reoptimizing = false;
    uint32_t CurVersion = 0;
  };

  using SPSReoptimizeArgList =
      shared::SPSArgList<ReOptMaterializationUnitID, uint32_t>;
  using SendErrorFn = unique_function<void(Error)>;

  Expected<SymbolMap> emitMUImplSymbols(ReOptMaterializationUnitState &MUState,
                                        uint32_t Version, JITDylib &JD,
                                        ThreadSafeModule TSM);

  void rt_reoptimize(SendErrorFn SendResult, ReOptMaterializationUnitID MUID,
                     uint32_t CurVersion);

  static Expected<Constant *>
  createReoptimizeArgBuffer(Module &M, ReOptMaterializationUnitID MUID,
                            uint32_t CurVersion);

  ReOptMaterializationUnitState &
  createMaterializationUnitState(const ThreadSafeModule &TSM);

  void
  registerMaterializationUnitResource(ResourceKey Key,
                                      ReOptMaterializationUnitState &State);

  ReOptMaterializationUnitState &
  getMaterializationUnitState(ReOptMaterializationUnitID MUID);

  ExecutionSession &ES;
  MangleAndInterner Mangle;
  IRLayer &BaseLayer;
  RedirectableSymbolManager &RSManager;

  ReOptimizeFunc ReOptFunc;
  AddProfilerFunc ProfilerFunc;

  std::mutex Mutex;
  std::map<ReOptMaterializationUnitID, ReOptMaterializationUnitState> MUStates;
  DenseMap<ResourceKey, DenseSet<ReOptMaterializationUnitID>> MUResources;
  ReOptMaterializationUnitID NextID = 1;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_REOPTIMIZELAYER_H
