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

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
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

namespace detail {

struct InjectionPointMixin {};

template <typename PassT, typename IRUnitT>
using HasRunOnIRUnit = decltype(std::declval<PassT>().run(
    std::declval<IRUnitT &>(), std::declval<AnalysisManager<IRUnitT> &>()));
template <typename PassT>
using HasRunOnLoop = decltype(std::declval<PassT>().run(
    std::declval<Loop &>(), std::declval<LoopAnalysisManager &>(),
    std::declval<LoopStandardAnalysisResults &>(),
    std::declval<LPMUpdater &>()));
template <typename PassT>
static constexpr bool isModulePass =
    is_detected<HasRunOnIRUnit, PassT, Module>::value;
template <typename PassT>
static constexpr bool isFunctionPass =
    is_detected<HasRunOnIRUnit, PassT, Function>::value;
template <typename PassT>
static constexpr bool isLoopPass = is_detected<HasRunOnLoop, PassT>::value;
template <typename PassT>
static constexpr bool isMachineFunctionPass =
    is_detected<HasRunOnIRUnit, PassT, MachineFunction>::value;

struct PassWrapper {
  StringRef Name;
  bool InCGSCC;
  bool IsInjectionPoint;

  template <typename PassT,
            typename = std::enable_if_t<
                isModulePass<PassT> || isFunctionPass<PassT> ||
                isLoopPass<PassT> || isMachineFunctionPass<PassT>>>
  PassWrapper(PassT &&P, bool InCGSCC = false)
      : Name(std::remove_reference_t<PassT>::name()), InCGSCC(InCGSCC) {
    if constexpr (isModulePass<PassT>) {
      Ctor.emplace<llvm::unique_function<void(ModulePassManager &)>>(
          [P = std::forward<PassT>(P)](ModulePassManager &PM) mutable {
            PM.addPass(std::move(P));
          });
    } else if constexpr (isFunctionPass<PassT>) {
      Ctor.emplace<llvm::unique_function<void(FunctionPassManager &)>>(
          [P = std::forward<PassT>(P)](FunctionPassManager &PM) mutable {
            PM.addPass(std::move(P));
          });
    } else if constexpr (isLoopPass<PassT>) {
      Ctor.emplace<llvm::unique_function<void(LoopPassManager &)>>(
          [P = std::forward<PassT>(P)](LoopPassManager &PM) mutable {
            PM.addPass(std::move(P));
          });
    } else if constexpr (isMachineFunctionPass<PassT>) {
      Ctor.emplace<llvm::unique_function<void(MachineFunctionPassManager &)>>(
          [P = std::forward<PassT>(P)](MachineFunctionPassManager &PM) mutable {
            PM.addPass(std::move(P));
          });
    }
    IsInjectionPoint =
        std::is_base_of_v<InjectionPointMixin, std::remove_reference_t<PassT>>;
  }

  std::variant<llvm::unique_function<void(ModulePassManager &)>,
               llvm::unique_function<void(FunctionPassManager &)>,
               llvm::unique_function<void(LoopPassManager &)>,
               llvm::unique_function<void(MachineFunctionPassManager &)>>
      Ctor;
};
} // namespace detail

class TargetMachineFunctionPassManager {
  friend class TargetFunctionPassManager;
  friend class TargetPassBuilder;

public:
  template <typename PassT>
  TargetMachineFunctionPassManager &addPass(PassT &&P) & {
    static_assert(detail::isMachineFunctionPass<PassT>,
                  "Not a machine function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  template <typename PassT>
  TargetMachineFunctionPassManager &&addPass(PassT &&P) && {
    static_assert(detail::isMachineFunctionPass<PassT>,
                  "Not a machine function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return std::move(*this);
  }

  TargetMachineFunctionPassManager &
  addPass(TargetMachineFunctionPassManager &&PM) & {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  TargetMachineFunctionPassManager &&
  addPass(TargetMachineFunctionPassManager &&PM) && {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return std::move(*this);
  }

private:
  std::list<detail::PassWrapper> Passes;
};

class TargetLoopPassManager {
  friend class TargetFunctionPassManager;
  friend class TargetPassBuilder;

public:
  template <typename PassT> TargetLoopPassManager &addPass(PassT &&P) & {
    static_assert(detail::isLoopPass<PassT>, "Not a loop pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  template <typename PassT> TargetLoopPassManager &&addPass(PassT &&P) && {
    static_assert(detail::isLoopPass<PassT>, "Not a loop pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return std::move(*this);
  }

  TargetLoopPassManager &addPass(TargetLoopPassManager &&PM) & {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  TargetLoopPassManager &&addPass(TargetLoopPassManager &&PM) && {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return std::move(*this);
  }

private:
  std::list<detail::PassWrapper> Passes;
};

class TargetFunctionPassManager {
  friend class TargetModulePassManager;
  friend class TargetPassBuilder;

public:
  template <typename PassT> TargetFunctionPassManager &addPass(PassT &&P) & {
    static_assert(detail::isFunctionPass<PassT>, "Not a function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  template <typename PassT> TargetFunctionPassManager &addPass(PassT &&P) && {
    static_assert(detail::isFunctionPass<PassT>, "Not a function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  TargetFunctionPassManager &addPass(TargetFunctionPassManager &&PM) & {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  TargetFunctionPassManager &&addPass(TargetFunctionPassManager &&PM) && {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return std::move(*this);
  }

  template <typename PassT>
  TargetFunctionPassManager &addLoopPass(PassT &&P) & {
    static_assert(detail::isLoopPass<PassT>, "Not a loop pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  template <typename PassT>
  TargetFunctionPassManager &&addLoopPass(PassT &&P) && {
    static_assert(detail::isLoopPass<PassT>, "Not a loop pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return std::move(*this);
  }

  TargetFunctionPassManager &addLoopPass(TargetLoopPassManager &&PM) & {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  TargetFunctionPassManager &&addLoopPass(TargetLoopPassManager &&PM) && {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return std::move(*this);
  }

  template <typename PassT>
  TargetFunctionPassManager &addMachineFunctionPass(PassT &&P) & {
    static_assert(detail::isMachineFunctionPass<PassT>,
                  "Not a machine function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  template <typename PassT>
  TargetFunctionPassManager &&addMachineFunctionPass(PassT &&P) && {
    static_assert(detail::isMachineFunctionPass<PassT>,
                  "Not a machine function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return std::move(*this);
  }

  TargetFunctionPassManager &
  addMachineFunctionPass(TargetMachineFunctionPassManager &&PM) & {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  TargetFunctionPassManager &&
  addMachineFunctionPass(TargetMachineFunctionPassManager &&PM) && {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return std::move(*this);
  }

private:
  std::list<detail::PassWrapper> Passes;
};

class TargetModulePassManager {
  friend class TargetPassBuilder;

public:
  template <typename PassT> TargetModulePassManager &addPass(PassT &&P) {
    static_assert(detail::isModulePass<PassT>, "Not a module pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  TargetModulePassManager &addPass(TargetModulePassManager &&PM) {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  template <typename PassT>
  TargetModulePassManager &addFunctionPass(PassT &&P) {
    static_assert(detail::isFunctionPass<PassT>, "Not a function pass!");
    Passes.emplace_back(std::forward<PassT>(P));
    return *this;
  }

  TargetModulePassManager &addFunctionPass(TargetFunctionPassManager &&PM) {
    for (auto &P : PM.Passes)
      Passes.push_back(std::move(P));
    PM.Passes.clear();
    return *this;
  }

  template <typename PassT>
  TargetModulePassManager &addFunctionPassWithPostOrderCGSCC(PassT &&P) {
    static_assert(detail::isFunctionPass<PassT>, "Not a function pass!");
    Passes.emplace_back(std::forward<PassT>(P), /*InCGSCC=*/true);
    return *this;
  }

  TargetModulePassManager &
  addFunctionPassWithPostOrderCGSCC(TargetFunctionPassManager &&PM) & {
    for (auto &P : PM.Passes) {
      P.InCGSCC = true;
      Passes.push_back(std::move(P));
    }
    PM.Passes.clear();
    return *this;
  }

  TargetModulePassManager &&
  addFunctionPassWithPostOrderCGSCC(TargetFunctionPassManager &&PM) && {
    for (auto &P : PM.Passes) {
      P.InCGSCC = true;
      Passes.push_back(std::move(P));
    }
    PM.Passes.clear();
    return std::move(*this);
  }

private:
  std::list<detail::PassWrapper> Passes;
};

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

  using PassList = std::list<detail::PassWrapper>;

protected:
  virtual void registerCallbacks() = 0;

  struct DummyFunctionPassMixin {
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {
      return PreservedAnalyses::all();
    }
  };
  struct DummyMachineFunctionPassMixin {
    PreservedAnalyses run(MachineFunction &, MachineFunctionAnalysisManager &) {
      return PreservedAnalyses::all();
    }
  };

  struct PreISelInjectionPoint : PassInfoMixin<PreISelInjectionPoint>,
                                 DummyFunctionPassMixin,
                                 detail::InjectionPointMixin {};

  struct PostBBSectionsInjectionPoint
      : PassInfoMixin<PostBBSectionsInjectionPoint>,
        DummyMachineFunctionPassMixin,
        detail::InjectionPointMixin {};

  struct PreEmitInjectionPoint : PassInfoMixin<PreEmitInjectionPoint>,
                                 DummyMachineFunctionPassMixin,
                                 detail::InjectionPointMixin {};

  struct ILPOptsInjectionPoint : PassInfoMixin<ILPOptsInjectionPoint>,
                                 DummyMachineFunctionPassMixin,
                                 detail::InjectionPointMixin {};

  PassBuilder &PB;
  TargetMachine *TM;
  CodeGenOptLevel OptLevel;
  CGPassBuilderOption CGPBO = getCGPassBuilderOption();

  /// @brief Add custom passes at injection point PassT
  /// @tparam PassT The injection point
  /// @tparam This is the recommended approach to extend the pass pipeline.
  /// PassManagerT Returned pass manager, by default it depends on the injection
  /// point.
  /// @param F Callback to build the pipeline.
  template <
      typename PassT,
      typename PassManagerT = std::conditional_t<
          isModulePass<PassT>, TargetModulePassManager,
          std::conditional_t<isFunctionPass<PassT>, TargetFunctionPassManager,
                             TargetMachineFunctionPassManager>>>
  void injectAt(
      typename llvm::type_identity<std::function<PassManagerT()>>::type F) {
    static_assert(std::is_base_of_v<detail::InjectionPointMixin, PassT>,
                  "Only injection points are supported!");
    injectBefore<PassT, PassManagerT>(F);
  }

  /// @brief Add custom passes before PassT
  /// @tparam PassT The injection point
  /// @tparam PassManagerT Returned pass manager, by default it depends on the
  /// injection point. Use this method only when there is no appropriate
  /// injection point.
  /// @param F Callback to build the pipeline.
  template <
      typename PassT,
      typename PassManagerT = std::conditional_t<
          isModulePass<PassT>, TargetModulePassManager,
          std::conditional_t<isFunctionPass<PassT>, TargetFunctionPassManager,
                             TargetMachineFunctionPassManager>>>
  void injectBefore(
      typename llvm::type_identity<std::function<PassManagerT()>>::type F) {
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

  /// @brief Add custom passes after PassT
  /// @tparam PassT The injection point
  /// @tparam PassManagerT Returned pass manager, by default it depends on the
  /// injection point. Use this method only when there is no appropriate
  /// injection point.
  /// @param F Callback to build the pipeline.
  template <
      typename PassT,
      typename PassManagerT = std::conditional_t<
          isModulePass<PassT>, TargetModulePassManager,
          std::conditional_t<isFunctionPass<PassT>, TargetFunctionPassManager,
                             TargetMachineFunctionPassManager>>>
  void injectAfter(
      typename llvm::type_identity<std::function<PassManagerT()>>::type F) {
    InjectionCallbacks.push_back(
        [Accessed = false, F](PassList &Passes, PassList::iterator I) mutable {
          if (Accessed)
            return I;
          if (PassT::name() != I->Name)
            return I;
          Accessed = true;
          auto PMPasses = F().Passes;
          auto NextI = std::next(I);
          return Passes.insert(NextI, std::make_move_iterator(PMPasses.begin()),
                               std::make_move_iterator(PMPasses.end()));
        });
  }

  /// @brief Register selection dag isel pass
  /// @tparam BuilderFuncT
  /// @param F A function returns a selection dag isel pass.
  template <typename BuilderFuncT>
  void registerSelectionDAGISelPass(BuilderFuncT F) {
    AddSelectionDAGISelPass = [=](TargetMachineFunctionPassManager &MFPM) {
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
  void buildCoreCodeGenPipeline(TargetModulePassManager &MPM);

  TargetModulePassManager buildCodeGenIRPipeline();

  /// Add passes that optimize machine instructions in SSA form.
  void addISelPasses(TargetMachineFunctionPassManager &MFPM);
  void addMachineSSAOptimizationPasses(TargetMachineFunctionPassManager &MFPM);
  void addRegAllocPipeline(TargetMachineFunctionPassManager &MFPM);
  void addRegAllocPass(TargetMachineFunctionPassManager &MFPM, bool Optimized);
  TargetModulePassManager buildCodeGenMIRPipeline();

  void addExceptionHandlingPasses(TargetFunctionPassManager &FPM);

  void filterPassList(TargetModulePassManager &MPM) const;

  void addPrinterPassesAndFreeMachineFunction(TargetModulePassManager &MPM,
                                              raw_pwrite_stream &Out,
                                              raw_pwrite_stream *DwoOut,
                                              CodeGenFileType FileType,
                                              MCContext &Ctx);

  ModulePassManager
  constructRealPassManager(TargetModulePassManager &MPMW) const;

private:
  virtual void anchor();

  StringSet<> DisabedPasses;
  std::vector<std::function<PassList::iterator(PassList &, PassList::iterator)>>
      InjectionCallbacks;
  std::function<void(TargetMachineFunctionPassManager &)>
      AddSelectionDAGISelPass;

  void invokeInjectionCallbacks(TargetModulePassManager &MPM) const;
};

} // namespace llvm

#endif
