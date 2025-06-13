//===- Parsing, selection, and construction of pass pipelines --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for registering analysis passes, producing common pass manager
/// configurations, and parsing of pass pipelines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_TARGETPASSBUILDER_H
#define LLVM_PASSES_TARGETPASSBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/identity.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include <list>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

class PassBuilder;
class TargetMachine;
class SelectionDAGISel;

/// @brief Build CodeGen pipeline
///
class TargetPassBuilder {
public:
  TargetPassBuilder(PassBuilder &PB);

  virtual ~TargetPassBuilder() = default;

  // TODO: Add necessary parameters once AsmPrinter is ported to new pass
  // manager.
  llvm::ModulePassManager buildPipeline(raw_pwrite_stream &Out,
                                        raw_pwrite_stream *DwoOut,
                                        CodeGenFileType FileType,
                                        MCContext &Ctx);

private:
  struct PassWrapper {
    StringRef Name;
    std::variant<llvm::ModulePassManager, llvm::FunctionPassManager,
                 llvm::LoopPassManager, llvm::MachineFunctionPassManager>
        InternalPass;
    bool InCGSCC = false;

    template <typename PassManagerT>
    PassWrapper(StringRef Name, PassManagerT &&PM)
        : Name(Name), InternalPass(std::forward<PassManagerT>(PM)) {}

    template <typename PassT> PassWrapper(PassT &&P) : Name(PassT::name()) {
      // FIXME: This can't handle the case when `run` is template.
      if constexpr (isModulePass<PassT>) {
        llvm::ModulePassManager MPM;
        MPM.addPass(std::forward<PassT>(P));
        InternalPass.emplace<llvm::ModulePassManager>(std::move(MPM));
      } else if constexpr (isFunctionPass<PassT>) {
        llvm::FunctionPassManager FPM;
        FPM.addPass(std::forward<PassT>(P));
        InternalPass.emplace<llvm::FunctionPassManager>(std::move(FPM));
      } else {
        static_assert(isMachineFunctionPass<PassT>, "Invalid pass type!");
        llvm::MachineFunctionPassManager MFPM;
        MFPM.addPass(std::forward<PassT>(P));
        InternalPass.emplace<llvm::MachineFunctionPassManager>(std::move(MFPM));
      }
    }
  };

public:
  using PassList = std::list<PassWrapper>;

private:
  template <typename InternalPassT> struct AdaptorWrapper : InternalPassT {
    using InternalPassT::Passes;
  };

  template <typename PassManagerT, typename InternalPassT = void>
  class PassManagerWrapper {
    friend class TargetPassBuilder;

  public:
    bool isEmpty() const { return Passes.empty(); }

    template <typename PassT> void addPass(PassT &&P) {
      PassManagerT PM;
      PM.addPass(std::forward<PassT>(P));
      // Injection point doesn't add real pass.
      if constexpr (std::is_base_of_v<InjectionPointMixin, PassT>)
        PM = PassManagerT();
      PassWrapper PW(PassT::name(), std::move(PM));
      Passes.push_back(std::move(PW));
    }

    void addPass(PassManagerWrapper &&PM) {
      for (auto &P : PM.Passes)
        Passes.push_back(std::move(P));
    }

    void addPass(AdaptorWrapper<InternalPassT> &&Adaptor) {
      for (auto &P : Adaptor.Passes)
        Passes.push_back(std::move(P));
    }

    void addPass(llvm::ModulePassManager &&) = delete;
    void addPass(llvm::FunctionPassManager &&) = delete;
    void addPass(llvm::LoopPassManager &&) = delete;
    void addPass(llvm::MachineFunctionPassManager &&) = delete;

  private:
    PassList Passes;
  };

  template <typename NestedPassManagerT, typename PassT>
  AdaptorWrapper<NestedPassManagerT> createPassAdaptor(PassT &&P) {
    AdaptorWrapper<NestedPassManagerT> Adaptor;
    Adaptor.addPass(std::forward<PassT>(P));
    return Adaptor;
  }

private:
  template <typename PassT, typename IRUnitT>
  using HasRunOnIRUnit = decltype(std::declval<PassT>().run(
      std::declval<IRUnitT &>(), std::declval<AnalysisManager<IRUnitT> &>()));
  template <typename PassT>
  static constexpr bool isModulePass =
      is_detected<HasRunOnIRUnit, PassT, Module>::value;
  template <typename PassT>
  static constexpr bool isFunctionPass =
      is_detected<HasRunOnIRUnit, PassT, Function>::value;
  template <typename PassT>
  static constexpr bool isMachineFunctionPass =
      is_detected<HasRunOnIRUnit, PassT, MachineFunction>::value;

protected:
  // Hijack real pass managers intentionally.
  using MachineFunctionPassManager =
      PassManagerWrapper<llvm::MachineFunctionPassManager>;
  using FunctionPassManager =
      PassManagerWrapper<llvm::FunctionPassManager, MachineFunctionPassManager>;
  using ModulePassManager =
      PassManagerWrapper<llvm::ModulePassManager, FunctionPassManager>;

  struct CGSCCAdaptorWrapper : AdaptorWrapper<FunctionPassManager> {};

protected:
  template <typename FunctionPassT>
  AdaptorWrapper<FunctionPassManager>
  createModuleToFunctionPassAdaptor(FunctionPassT &&P) {
    return createPassAdaptor<FunctionPassManager>(
        std::forward<FunctionPassT>(P));
  }

  AdaptorWrapper<FunctionPassManager>
  createModuleToPostOrderCGSCCPassAdaptor(CGSCCAdaptorWrapper &&PM) {
    AdaptorWrapper<FunctionPassManager> AW;
    AW.Passes = std::move(PM.Passes);
    return AW;
  }

  template <typename FunctionPassT>
  CGSCCAdaptorWrapper createCGSCCToFunctionPassAdaptor(FunctionPassT &&PM) {
    for (auto &P : PM.Passes)
      P.InCGSCC = true;
    CGSCCAdaptorWrapper AW;
    AW.Passes = std::move(PM.Passes);
    return AW;
  }

  template <typename MachineFunctionPassT>
  AdaptorWrapper<MachineFunctionPassManager>
  createFunctionToMachineFunctionPassAdaptor(MachineFunctionPassT &&P) {
    return createPassAdaptor<MachineFunctionPassManager>(
        std::forward<MachineFunctionPassT>(P));
  }

  struct InjectionPointMixin {};
  // When run is template, injectBefore can't recognize pass type correctly.
  struct DummyFunctionPassBase : InjectionPointMixin {
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {
      return PreservedAnalyses();
    }
  };
  struct DummyMachineFunctionPassBase : InjectionPointMixin {
    PreservedAnalyses run(MachineFunction &, MachineFunctionAnalysisManager &) {
      return PreservedAnalyses();
    }
  };
  struct PreISel : PassInfoMixin<PreISel>, DummyFunctionPassBase {};
  struct PostBBSections : PassInfoMixin<PostBBSections>,
                          DummyMachineFunctionPassBase {};
  struct PreEmit : PassInfoMixin<PreEmit>, DummyMachineFunctionPassBase {};

protected:
  PassBuilder &PB;
  TargetMachine *TM;
  CodeGenOptLevel OptLevel;
  CGPassBuilderOption CGPBO = getCGPassBuilderOption();

  /// @brief The only method to extend pipeline
  /// @tparam PassT The injection point
  /// @tparam PassManagerT Returned pass manager, by default it depends on the
  /// injection point.
  /// @param F Callback to build the pipeline.
  template <typename PassT,
            typename PassManagerT = std::conditional_t<
                isModulePass<PassT>, ModulePassManager,
                std::conditional_t<isFunctionPass<PassT>, FunctionPassManager,
                                   MachineFunctionPassManager>>>
  void injectBefore(
      typename llvm::identity<std::function<PassManagerT()>>::argument_type F) {
    InjectionCallbacks.push_back(
        [Accessed = false, F](PassList &Passes, PassList::iterator I) mutable {
          if (Accessed)
            return I;
          if (PassT::name() != I->Name)
            return I;
          Accessed = true;
          auto PMPasses = F().Passes;
          return Passes.insert(I, std::make_move_iterator(PMPasses.begin()),
                               std::make_move_iterator(PMPasses.end()));
        });
  }

  /// @brief Register selection dag isel pass
  /// @tparam BuilderFuncT
  /// @param F A function returns a selection dag isel pass.
  template <typename BuilderFuncT>
  void registerSelectionDAGISelPass(BuilderFuncT F) {
    AddSelectionDAGISelPass = [=](MachineFunctionPassManager &MFPM) {
      using ResultT = std::invoke_result_t<BuilderFuncT>;
      static_assert(isMachineFunctionPass<ResultT> &&
                        !std::is_same_v<MachineFunctionPassManager, ResultT>,
                    "Please add only SelectionDAGISelPass!");
      MFPM.addPass(F());
    };
  }

  template <typename PassTs> void disablePass() {
    DisabedPasses.insert(PassTs::name());
  }

  void disablePass(StringRef Name) { DisabedPasses.insert(Name); }

  template <typename PassT> bool isPassDisabled() const {
    return DisabedPasses.contains(PassT::name());
  }

  bool isPassDisabled(StringRef Name) const {
    return DisabedPasses.contains(Name);
  }

  template <typename PassT> bool isPassEnabled() const {
    return !isPassDisabled<PassT>();
  }

  bool isPassEnabled(StringRef Name) const { return !isPassDisabled(Name); }

private:
  void buildCoreCodeGenPipeline(ModulePassManager &MPM);

  ModulePassManager buildCodeGenIRPipeline();

  /// Add passes that optimize machine instructions in SSA form.
  void addISelPasses(MachineFunctionPassManager &MFPM);
  void addMachineSSAOptimizationPasses(MachineFunctionPassManager &MFPM);
  void addRegAllocPipeline(MachineFunctionPassManager &MFPM);
  void addRegAllocPass(MachineFunctionPassManager &MFPM, bool Optimized);
  ModulePassManager buildCodeGenMIRPipeline();

  void addExceptionHandlingPasses(FunctionPassManager &FPM);

  void filtPassList(ModulePassManager &MPM) const;

  void addPrinterPassesAndFreeMachineFunction(ModulePassManager &MPM,
                                              raw_pwrite_stream &Out,
                                              raw_pwrite_stream *DwoOut,
                                              CodeGenFileType FileType,
                                              MCContext &Ctx);

  llvm::ModulePassManager constructRealPassManager(ModulePassManager &&MPMW);

private:
  virtual void anchor();

  StringSet<> DisabedPasses;
  std::vector<std::function<PassList::iterator(PassList &, PassList::iterator)>>
      InjectionCallbacks;
  std::function<void(MachineFunctionPassManager &)> AddSelectionDAGISelPass;

  void invokeInjectionCallbacks(ModulePassManager &MPM) const;

  // Only Loop Strength Reduction need this, shadow LoopPassManager
  // in future if it is necessary.
  template <typename PassT>
  void addLoopPass(FunctionPassManager &FPM, PassT &&P) {
    LoopPassManager LPM;
    LPM.addPass(std::forward<PassT>(P));
    FPM.Passes.push_back(PassWrapper(PassT::name(), std::move(LPM)));
  }
};

template <> struct TargetPassBuilder::AdaptorWrapper<void> {};

} // namespace llvm

#endif
