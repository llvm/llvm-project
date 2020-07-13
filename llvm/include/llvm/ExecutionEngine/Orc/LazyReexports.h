//===------ LazyReexports.h -- Utilities for lazy reexports -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lazy re-exports are similar to normal re-exports, except that for callable
// symbols the definitions are replaced with trampolines that will look up and
// call through to the re-exported symbol at runtime. This can be used to
// enable lazy compilation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LAZYREEXPORTS_H
#define LLVM_EXECUTIONENGINE_ORC_LAZYREEXPORTS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/Speculation.h"

namespace llvm {

class Triple;

namespace orc {

/// Manages a set of 'lazy call-through' trampolines. These are compiler
/// re-entry trampolines that are pre-bound to look up a given symbol in a given
/// JITDylib, then jump to that address. Since compilation of symbols is
/// triggered on first lookup, these call-through trampolines can be used to
/// implement lazy compilation.
///
/// The easiest way to construct these call-throughs is using the lazyReexport
/// function.
class LazyCallThroughManager {
public:
  using NotifyResolvedFunction =
      unique_function<Error(JITTargetAddress ResolvedAddr)>;

  // Return a free call-through trampoline and bind it to look up and call
  // through to the given symbol.
  Expected<JITTargetAddress>
  getCallThroughTrampoline(JITDylib &SourceJD, SymbolStringPtr SymbolName,
                           NotifyResolvedFunction NotifyResolved);

  void resolveTrampolineLandingAddress(
      JITTargetAddress TrampolineAddr,
      TrampolinePool::NotifyLandingResolvedFunction NotifyLandingResolved);

protected:
  using NotifyLandingResolvedFunction =
      TrampolinePool::NotifyLandingResolvedFunction;

  LazyCallThroughManager(ExecutionSession &ES,
                         JITTargetAddress ErrorHandlerAddr,
                         std::unique_ptr<TrampolinePool> TP);

  struct ReexportsEntry {
    JITDylib *SourceJD;
    SymbolStringPtr SymbolName;
  };

  JITTargetAddress reportCallThroughError(Error Err);
  Expected<ReexportsEntry> findReexport(JITTargetAddress TrampolineAddr);
  Error notifyResolved(JITTargetAddress TrampolineAddr,
                       JITTargetAddress ResolvedAddr);
  void setTrampolinePool(std::unique_ptr<TrampolinePool> TP) {
    this->TP = std::move(TP);
  }

private:
  using ReexportsMap = std::map<JITTargetAddress, ReexportsEntry>;

  using NotifiersMap = std::map<JITTargetAddress, NotifyResolvedFunction>;

  std::mutex LCTMMutex;
  ExecutionSession &ES;
  JITTargetAddress ErrorHandlerAddr;
  std::unique_ptr<TrampolinePool> TP;
  ReexportsMap Reexports;
  NotifiersMap Notifiers;
};

/// A lazy call-through manager that builds trampolines in the current process.
class LocalLazyCallThroughManager : public LazyCallThroughManager {
private:
  using NotifyTargetResolved = unique_function<void(JITTargetAddress)>;

  LocalLazyCallThroughManager(ExecutionSession &ES,
                              JITTargetAddress ErrorHandlerAddr)
      : LazyCallThroughManager(ES, ErrorHandlerAddr, nullptr) {}

  template <typename ORCABI> Error init() {
    auto TP = LocalTrampolinePool<ORCABI>::Create(
        [this](JITTargetAddress TrampolineAddr,
               TrampolinePool::NotifyLandingResolvedFunction
                   NotifyLandingResolved) {
          resolveTrampolineLandingAddress(TrampolineAddr,
                                          std::move(NotifyLandingResolved));
        });

    if (!TP)
      return TP.takeError();

    setTrampolinePool(std::move(*TP));
    return Error::success();
  }

public:
  /// Create a LocalLazyCallThroughManager using the given ABI. See
  /// createLocalLazyCallThroughManager.
  template <typename ORCABI>
  static Expected<std::unique_ptr<LocalLazyCallThroughManager>>
  Create(ExecutionSession &ES, JITTargetAddress ErrorHandlerAddr) {
    auto LLCTM = std::unique_ptr<LocalLazyCallThroughManager>(
        new LocalLazyCallThroughManager(ES, ErrorHandlerAddr));

    if (auto Err = LLCTM->init<ORCABI>())
      return std::move(Err);

    return std::move(LLCTM);
  }
};

/// Create a LocalLazyCallThroughManager from the given triple and execution
/// session.
Expected<std::unique_ptr<LazyCallThroughManager>>
createLocalLazyCallThroughManager(const Triple &T, ExecutionSession &ES,
                                  JITTargetAddress ErrorHandlerAddr);

/// A materialization unit that builds lazy re-exports. These are callable
/// entry points that call through to the given symbols.
/// Unlike a 'true' re-export, the address of the lazy re-export will not
/// match the address of the re-exported symbol, but calling it will behave
/// the same as calling the re-exported symbol.
class LazyReexportsMaterializationUnit : public MaterializationUnit {
public:
  LazyReexportsMaterializationUnit(LazyCallThroughManager &LCTManager,
                                   IndirectStubsManager &ISManager,
                                   JITDylib &SourceJD,
                                   SymbolAliasMap CallableAliases,
                                   ImplSymbolMap *SrcJDLoc, VModuleKey K);

  StringRef getName() const override;

private:
  void materialize(MaterializationResponsibility R) override;
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override;
  static SymbolFlagsMap extractFlags(const SymbolAliasMap &Aliases);

  LazyCallThroughManager &LCTManager;
  IndirectStubsManager &ISManager;
  JITDylib &SourceJD;
  SymbolAliasMap CallableAliases;
  ImplSymbolMap *AliaseeTable;
};

/// Define lazy-reexports based on the given SymbolAliasMap. Each lazy re-export
/// is a callable symbol that will look up and dispatch to the given aliasee on
/// first call. All subsequent calls will go directly to the aliasee.
inline std::unique_ptr<LazyReexportsMaterializationUnit>
lazyReexports(LazyCallThroughManager &LCTManager,
              IndirectStubsManager &ISManager, JITDylib &SourceJD,
              SymbolAliasMap CallableAliases, ImplSymbolMap *SrcJDLoc = nullptr,
              VModuleKey K = VModuleKey()) {
  return std::make_unique<LazyReexportsMaterializationUnit>(
      LCTManager, ISManager, SourceJD, std::move(CallableAliases), SrcJDLoc,
      std::move(K));
}

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LAZYREEXPORTS_H
