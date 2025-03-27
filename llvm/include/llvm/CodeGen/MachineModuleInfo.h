//===-- llvm/CodeGen/MachineModuleInfo.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collect meta information for a module.  This information should be in a
// neutral form that can be used by different debugging and exception handling
// schemes.
//
// The organization of information is primarily clustered around the source
// compile units.  The main exception is source line correspondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineModuleInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)
//  -- Source line correspondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     emitter to generate labels referenced by debug information tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFO_H
#define LLVM_CODEGEN_MACHINEMODULEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Pass.h"
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

class Function;
class TargetMachine;
class MachineFunction;
class Module;

//===----------------------------------------------------------------------===//
/// This class can be derived from and used by targets to hold private
/// target-specific information for each Module.  Objects of type are
/// accessed/created with MachineModuleInfo::getObjFileInfo and destroyed when
/// the MachineModuleInfo is destroyed.
///
class MachineModuleInfoImpl {
public:
  using StubValueTy = PointerIntPair<MCSymbol *, 1, bool>;
  using SymbolListTy = std::vector<std::pair<MCSymbol *, StubValueTy>>;

  /// A variant of SymbolListTy where the stub is a generalized MCExpr.
  using ExprStubListTy = std::vector<std::pair<MCSymbol *, const MCExpr *>>;

  virtual ~MachineModuleInfoImpl();

protected:
  /// Return the entries from a DenseMap in a deterministic sorted orer.
  /// Clears the map.
  static SymbolListTy getSortedStubs(DenseMap<MCSymbol*, StubValueTy>&);

  /// Return the entries from a DenseMap in a deterministic sorted orer.
  /// Clears the map.
  static ExprStubListTy
  getSortedExprStubs(DenseMap<MCSymbol *, const MCExpr *> &);
};

//===----------------------------------------------------------------------===//
/// This class contains meta information specific to a module.  Queries can be
/// made by different debugging and exception handling schemes and reformated
/// for specific use.
///
class MachineModuleInfo {
  friend class MachineModuleInfoWrapperPass;
  friend class MachineModuleAnalysis;

  const TargetMachine &TM;

  /// This is the MCContext used for the entire code generator.
  MCContext Context;
  // This is an external context, that if assigned, will be used instead of the
  // internal context.
  MCContext *ExternalContext = nullptr;

  /// This is the LLVM Module being worked on.
  const Module *TheModule = nullptr;

  /// This is the object-file-format-specific implementation of
  /// MachineModuleInfoImpl, which lets targets accumulate whatever info they
  /// want.
  MachineModuleInfoImpl *ObjFileMMI;

  /// Maps IR Functions to their corresponding MachineFunctions.
  DenseMap<const Function*, std::unique_ptr<MachineFunction>> MachineFunctions;
  /// Next unique number available for a MachineFunction.
  unsigned NextFnNum = 0;
  const Function *LastRequest = nullptr; ///< Used for shortcut/cache.
  MachineFunction *LastResult = nullptr; ///< Used for shortcut/cache.

  MachineModuleInfo &operator=(MachineModuleInfo &&MMII) = delete;

public:
  explicit MachineModuleInfo(const TargetMachine *TM = nullptr);

  explicit MachineModuleInfo(const TargetMachine *TM, MCContext *ExtContext);

  MachineModuleInfo(MachineModuleInfo &&MMII);

  ~MachineModuleInfo();

  void initialize();
  void finalize();

  const TargetMachine &getTarget() const { return TM; }

  const MCContext &getContext() const {
    return ExternalContext ? *ExternalContext : Context;
  }
  MCContext &getContext() {
    return ExternalContext ? *ExternalContext : Context;
  }

  const Module *getModule() const { return TheModule; }

  /// Returns the MachineFunction constructed for the IR function \p F.
  /// Creates a new MachineFunction if none exists yet.
  /// NOTE: New pass manager clients shall not use this method to get
  /// the `MachineFunction`, use `MachineFunctionAnalysis` instead.
  MachineFunction &getOrCreateMachineFunction(Function &F);

  /// \brief Returns the MachineFunction associated to IR function \p F if there
  /// is one, otherwise nullptr.
  /// NOTE: New pass manager clients shall not use this method to get
  /// the `MachineFunction`, use `MachineFunctionAnalysis` instead.
  MachineFunction *getMachineFunction(const Function &F) const;

  /// Delete the MachineFunction \p MF and reset the link in the IR Function to
  /// Machine Function map.
  void deleteMachineFunctionFor(Function &F);

  /// Add an externally created MachineFunction \p MF for \p F.
  void insertFunction(const Function &F, std::unique_ptr<MachineFunction> &&MF);

  /// Keep track of various per-module pieces of information for backends
  /// that would like to do so.
  template<typename Ty>
  Ty &getObjFileInfo() {
    if (ObjFileMMI == nullptr)
      ObjFileMMI = new Ty(*this);
    return *static_cast<Ty*>(ObjFileMMI);
  }

  template<typename Ty>
  const Ty &getObjFileInfo() const {
    return const_cast<MachineModuleInfo*>(this)->getObjFileInfo<Ty>();
  }

  /// \}
}; // End class MachineModuleInfo

class MachineModuleInfoWrapperPass : public ImmutablePass {
  MachineModuleInfo MMI;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit MachineModuleInfoWrapperPass(const TargetMachine *TM = nullptr);

  explicit MachineModuleInfoWrapperPass(const TargetMachine *TM,
                                        MCContext *ExtContext);

  // Initialization and Finalization
  bool doInitialization(Module &) override;
  bool doFinalization(Module &) override;

  MachineModuleInfo &getMMI() { return MMI; }
  const MachineModuleInfo &getMMI() const { return MMI; }
};

/// An analysis that produces \c MachineModuleInfo for a module.
/// This does not produce its own MachineModuleInfo because we need a consistent
/// MachineModuleInfo to keep ownership of MachineFunctions regardless of
/// analysis invalidation/clearing. So something outside the analysis
/// infrastructure must own the MachineModuleInfo.
class MachineModuleAnalysis : public AnalysisInfoMixin<MachineModuleAnalysis> {
  friend AnalysisInfoMixin<MachineModuleAnalysis>;
  static AnalysisKey Key;

  MachineModuleInfo &MMI;

public:
  class Result {
    MachineModuleInfo &MMI;
    Result(MachineModuleInfo &MMI) : MMI(MMI) {}
    friend class MachineModuleAnalysis;

  public:
    MachineModuleInfo &getMMI() { return MMI; }

    // MMI owes MCContext. It should never be invalidated.
    bool invalidate(Module &, const PreservedAnalyses &,
                    ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  MachineModuleAnalysis(MachineModuleInfo &MMI) : MMI(MMI) {}

  /// Run the analysis pass and produce machine module information.
  Result run(Module &M, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEMODULEINFO_H
