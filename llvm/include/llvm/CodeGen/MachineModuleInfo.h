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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Pass.h"
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

class BasicBlock;
class Function;
class LLVMTargetMachine;
class MMIAddrLabelMap;
class MachineFunction;
class Module;
class MCSymbol;

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

  virtual ~MachineModuleInfoImpl();

protected:
  /// Return the entries from a DenseMap in a deterministic sorted orer.
  /// Clears the map.
  static SymbolListTy getSortedStubs(DenseMap<MCSymbol*, StubValueTy>&);
};

//===----------------------------------------------------------------------===//
/// This class contains meta information specific to a module.  Queries can be
/// made by different debugging and exception handling schemes and reformated
/// for specific use.
///
class MachineModuleInfo {
  friend class MachineModuleInfoWrapperPass;
  friend class MachineModuleAnalysis;

  const LLVMTargetMachine &TM;

  /// This is the MCContext used for the entire code generator.
  MCContext Context;
  // This is an external context, that if assigned, will be used instead of the
  // internal context.
  MCContext *ExternalContext = nullptr;

  /// This is the LLVM Module being worked on.
  const Module *TheModule;

  /// This is the object-file-format-specific implementation of
  /// MachineModuleInfoImpl, which lets targets accumulate whatever info they
  /// want.
  MachineModuleInfoImpl *ObjFileMMI;

  /// \name Exception Handling
  /// \{

  /// Vector of all personality functions ever seen. Used to emit common EH
  /// frames.
  std::vector<const Function *> Personalities;

  /// The current call site index being processed, if any. 0 if none.
  unsigned CurCallSite;

  /// \}

  /// This map keeps track of which symbol is being used for the specified
  /// basic block's address of label.
  MMIAddrLabelMap *AddrLabelSymbols;

  // TODO: Ideally, what we'd like is to have a switch that allows emitting
  // synchronous (precise at call-sites only) CFA into .eh_frame. However,
  // even under this switch, we'd like .debug_frame to be precise when using
  // -g. At this moment, there's no way to specify that some CFI directives
  // go into .eh_frame only, while others go into .debug_frame only.

  /// True if debugging information is available in this module.
  bool DbgInfoAvailable;

  /// True if this module is being built for windows/msvc, and uses floating
  /// point.  This is used to emit an undefined reference to _fltused.
  bool UsesMSVCFloatingPoint;

  /// True if the module calls the __morestack function indirectly, as is
  /// required under the large code model on x86. This is used to emit
  /// a definition of a symbol, __morestack_addr, containing the address. See
  /// comments in lib/Target/X86/X86FrameLowering.cpp for more details.
  bool UsesMorestackAddr;

  /// Maps IR Functions to their corresponding MachineFunctions.
  DenseMap<const Function*, std::unique_ptr<MachineFunction>> MachineFunctions;
  /// Next unique number available for a MachineFunction.
  unsigned NextFnNum = 0;
  const Function *LastRequest = nullptr; ///< Used for shortcut/cache.
  MachineFunction *LastResult = nullptr; ///< Used for shortcut/cache.

  MachineModuleInfo &operator=(MachineModuleInfo &&MMII) = delete;

public:
  explicit MachineModuleInfo(const LLVMTargetMachine *TM = nullptr);

  explicit MachineModuleInfo(const LLVMTargetMachine *TM,
                             MCContext *ExtContext);

  MachineModuleInfo(MachineModuleInfo &&MMII);

  ~MachineModuleInfo();

  void initialize();
  void finalize();

  const LLVMTargetMachine &getTarget() const { return TM; }

  const MCContext &getContext() const {
    return ExternalContext ? *ExternalContext : Context;
  }
  MCContext &getContext() {
    return ExternalContext ? *ExternalContext : Context;
  }

  const Module *getModule() const { return TheModule; }

  /// Returns the MachineFunction constructed for the IR function \p F.
  /// Creates a new MachineFunction if none exists yet.
  MachineFunction &getOrCreateMachineFunction(Function &F);

  /// \brief Returns the MachineFunction associated to IR function \p F if there
  /// is one, otherwise nullptr.
  MachineFunction *getMachineFunction(const Function &F) const;

  /// Delete the MachineFunction \p MF and reset the link in the IR Function to
  /// Machine Function map.
  void deleteMachineFunctionFor(Function &F);

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

  /// Returns true if valid debug info is present.
  bool hasDebugInfo() const { return DbgInfoAvailable; }

  bool usesMSVCFloatingPoint() const { return UsesMSVCFloatingPoint; }

  void setUsesMSVCFloatingPoint(bool b) { UsesMSVCFloatingPoint = b; }

  bool usesMorestackAddr() const {
    return UsesMorestackAddr;
  }

  void setUsesMorestackAddr(bool b) {
    UsesMorestackAddr = b;
  }

  /// Return the symbol to be used for the specified basic block when its
  /// address is taken.  This cannot be its normal LBB label because the block
  /// may be accessed outside its containing function.
  MCSymbol *getAddrLabelSymbol(const BasicBlock *BB) {
    return getAddrLabelSymbolToEmit(BB).front();
  }

  /// Return the symbol to be used for the specified basic block when its
  /// address is taken.  If other blocks were RAUW'd to this one, we may have
  /// to emit them as well, return the whole set.
  ArrayRef<MCSymbol *> getAddrLabelSymbolToEmit(const BasicBlock *BB);

  /// If the specified function has had any references to address-taken blocks
  /// generated, but the block got deleted, return the symbol now so we can
  /// emit it.  This prevents emitting a reference to a symbol that has no
  /// definition.
  void takeDeletedSymbolsForFunction(const Function *F,
                                     std::vector<MCSymbol*> &Result);

  /// \name Exception Handling
  /// \{

  /// Set the call site currently being processed.
  void setCurrentCallSite(unsigned Site) { CurCallSite = Site; }

  /// Get the call site currently being processed, if any.  return zero if
  /// none.
  unsigned getCurrentCallSite() { return CurCallSite; }

  /// Provide the personality function for the exception information.
  void addPersonality(const Function *Personality);

  /// Return array of personality functions ever seen.
  const std::vector<const Function *>& getPersonalities() const {
    return Personalities;
  }
  /// \}

  // MMI owes MCContext. It should never be invalidated.
  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    return false;
  }
}; // End class MachineModuleInfo

class MachineModuleInfoWrapperPass : public ImmutablePass {
  MachineModuleInfo MMI;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit MachineModuleInfoWrapperPass(const LLVMTargetMachine *TM = nullptr);

  explicit MachineModuleInfoWrapperPass(const LLVMTargetMachine *TM,
                                        MCContext *ExtContext);

  // Initialization and Finalization
  bool doInitialization(Module &) override;
  bool doFinalization(Module &) override;

  MachineModuleInfo &getMMI() { return MMI; }
  const MachineModuleInfo &getMMI() const { return MMI; }
};

/// An analysis that produces \c MachineInfo for a module.
class MachineModuleAnalysis : public AnalysisInfoMixin<MachineModuleAnalysis> {
  friend AnalysisInfoMixin<MachineModuleAnalysis>;
  static AnalysisKey Key;

  const LLVMTargetMachine *TM;

public:
  /// Provide the result type for this analysis pass.
  using Result = MachineModuleInfo;

  MachineModuleAnalysis(const LLVMTargetMachine *TM) : TM(TM) {}

  /// Run the analysis pass and produce machine module information.
  MachineModuleInfo run(Module &M, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEMODULEINFO_H
