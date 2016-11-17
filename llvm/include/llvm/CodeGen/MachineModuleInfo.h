//===-- llvm/CodeGen/MachineModuleInfo.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class BlockAddress;
class CallInst;
class Constant;
class GlobalVariable;
class LandingPadInst;
class MDNode;
class MMIAddrLabelMap;
class MachineBasicBlock;
class MachineFunction;
class MachineFunctionInitializer;
class Module;
class PointerType;
class StructType;

struct SEHHandler {
  // Filter or finally function. Null indicates a catch-all.
  const Function *FilterOrFinally;

  // Address of block to recover at. Null for a finally handler.
  const BlockAddress *RecoverBA;
};

//===----------------------------------------------------------------------===//
/// This structure is used to retain landing pad info for the current function.
///
struct LandingPadInfo {
  MachineBasicBlock *LandingPadBlock;      // Landing pad block.
  SmallVector<MCSymbol *, 1> BeginLabels;  // Labels prior to invoke.
  SmallVector<MCSymbol *, 1> EndLabels;    // Labels after invoke.
  SmallVector<SEHHandler, 1> SEHHandlers;  // SEH handlers active at this lpad.
  MCSymbol *LandingPadLabel;               // Label at beginning of landing pad.
  std::vector<int> TypeIds;               // List of type ids (filters negative).

  explicit LandingPadInfo(MachineBasicBlock *MBB)
      : LandingPadBlock(MBB), LandingPadLabel(nullptr) {}
};

//===----------------------------------------------------------------------===//
/// This class can be derived from and used by targets to hold private
/// target-specific information for each Module.  Objects of type are
/// accessed/created with MMI::getInfo and destroyed when the MachineModuleInfo
/// is destroyed.
///
class MachineModuleInfoImpl {
public:
  typedef PointerIntPair<MCSymbol*, 1, bool> StubValueTy;
  virtual ~MachineModuleInfoImpl();
  typedef std::vector<std::pair<MCSymbol*, StubValueTy> > SymbolListTy;
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
class MachineModuleInfo : public ImmutablePass {
  const TargetMachine &TM;

  /// This is the MCContext used for the entire code generator.
  MCContext Context;

  /// This is the LLVM Module being worked on.
  const Module *TheModule;

  /// This is the object-file-format-specific implementation of
  /// MachineModuleInfoImpl, which lets targets accumulate whatever info they
  /// want.
  MachineModuleInfoImpl *ObjFileMMI;

  /// List of moves done by a function's prolog.  Used to construct frame maps
  /// by debug and exception handling consumers.
  std::vector<MCCFIInstruction> FrameInstructions;

  /// List of LandingPadInfo describing the landing pad information in the
  /// current function.
  std::vector<LandingPadInfo> LandingPads;

  /// Map a landing pad's EH symbol to the call site indexes.
  DenseMap<MCSymbol*, SmallVector<unsigned, 4> > LPadToCallSiteMap;

  /// Map of invoke call site index values to associated begin EH_LABEL for the
  /// current function.
  DenseMap<MCSymbol*, unsigned> CallSiteMap;

  /// The current call site index being processed, if any. 0 if none.
  unsigned CurCallSite;

  /// List of C++ TypeInfo used in the current function.
  std::vector<const GlobalValue *> TypeInfos;

  /// List of typeids encoding filters used in the current function.
  std::vector<unsigned> FilterIds;

  /// List of the indices in FilterIds corresponding to filter terminators.
  std::vector<unsigned> FilterEnds;

  /// Vector of all personality functions ever seen. Used to emit common EH
  /// frames.
  std::vector<const Function *> Personalities;

  /// This map keeps track of which symbol is being used for the specified
  /// basic block's address of label.
  MMIAddrLabelMap *AddrLabelSymbols;

  bool CallsEHReturn;
  bool CallsUnwindInit;
  bool HasEHFunclets;

  // TODO: Ideally, what we'd like is to have a switch that allows emitting 
  // synchronous (precise at call-sites only) CFA into .eh_frame. However,
  // even under this switch, we'd like .debug_frame to be precise when using.
  // -g. At this moment, there's no way to specify that some CFI directives
  // go into .eh_frame only, while others go into .debug_frame only.

  /// True if debugging information is available in this module.
  bool DbgInfoAvailable;

  /// True if this module calls VarArg function with floating-point arguments.
  /// This is used to emit an undefined reference to _fltused on Windows
  /// targets.
  bool UsesVAFloatArgument;

  /// True if the module calls the __morestack function indirectly, as is
  /// required under the large code model on x86. This is used to emit
  /// a definition of a symbol, __morestack_addr, containing the address. See
  /// comments in lib/Target/X86/X86FrameLowering.cpp for more details.
  bool UsesMorestackAddr;

  EHPersonality PersonalityTypeCache;

  MachineFunctionInitializer *MFInitializer;
  /// Maps IR Functions to their corresponding MachineFunctions.
  DenseMap<const Function*, std::unique_ptr<MachineFunction>> MachineFunctions;
  /// Next unique number available for a MachineFunction.
  unsigned NextFnNum = 0;
  const Function *LastRequest = nullptr; ///< Used for shortcut/cache.
  MachineFunction *LastResult = nullptr; ///< Used for shortcut/cache.

public:
  static char ID; // Pass identification, replacement for typeid

  struct VariableDbgInfo {
    const DILocalVariable *Var;
    const DIExpression *Expr;
    unsigned Slot;
    const DILocation *Loc;

    VariableDbgInfo(const DILocalVariable *Var, const DIExpression *Expr,
                    unsigned Slot, const DILocation *Loc)
        : Var(Var), Expr(Expr), Slot(Slot), Loc(Loc) {}
  };
  typedef SmallVector<VariableDbgInfo, 4> VariableDbgInfoMapTy;
  VariableDbgInfoMapTy VariableDbgInfos;

  explicit MachineModuleInfo(const TargetMachine *TM = nullptr);
  ~MachineModuleInfo() override;

  // Initialization and Finalization
  bool doInitialization(Module &) override;
  bool doFinalization(Module &) override;

  /// Discard function meta information.
  void EndFunction();

  const MCContext &getContext() const { return Context; }
  MCContext &getContext() { return Context; }

  void setModule(const Module *M) { TheModule = M; }
  const Module *getModule() const { return TheModule; }

  void setMachineFunctionInitializer(MachineFunctionInitializer *MFInit) {
    MFInitializer = MFInit;
  }

  /// Returns the MachineFunction constructed for the IR function \p F.
  /// Creates a new MachineFunction and runs the MachineFunctionInitializer
  /// if none exists yet.
  MachineFunction &getMachineFunction(const Function &F);

  /// Delete the MachineFunction \p MF and reset the link in the IR Function to
  /// Machine Function map.
  void deleteMachineFunctionFor(Function &F);

  /// Keep track of various per-function pieces of information for backends
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
  void setDebugInfoAvailability(bool avail) { DbgInfoAvailable = avail; }

  bool callsEHReturn() const { return CallsEHReturn; }
  void setCallsEHReturn(bool b) { CallsEHReturn = b; }

  bool callsUnwindInit() const { return CallsUnwindInit; }
  void setCallsUnwindInit(bool b) { CallsUnwindInit = b; }

  bool hasEHFunclets() const { return HasEHFunclets; }
  void setHasEHFunclets(bool V) { HasEHFunclets = V; }

  bool usesVAFloatArgument() const {
    return UsesVAFloatArgument;
  }

  void setUsesVAFloatArgument(bool b) {
    UsesVAFloatArgument = b;
  }

  bool usesMorestackAddr() const {
    return UsesMorestackAddr;
  }

  void setUsesMorestackAddr(bool b) {
    UsesMorestackAddr = b;
  }

  /// Returns a reference to a list of cfi instructions in the current
  /// function's prologue.  Used to construct frame maps for debug and
  /// exception handling comsumers.
  const std::vector<MCCFIInstruction> &getFrameInstructions() const {
    return FrameInstructions;
  }

  LLVM_NODISCARD unsigned addFrameInst(const MCCFIInstruction &Inst) {
    FrameInstructions.push_back(Inst);
    return FrameInstructions.size() - 1;
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


  //===- EH ---------------------------------------------------------------===//

  /// Find or create an LandingPadInfo for the specified MachineBasicBlock.
  LandingPadInfo &getOrCreateLandingPadInfo(MachineBasicBlock *LandingPad);

  /// Provide the begin and end labels of an invoke style call and associate it
  /// with a try landing pad block.
  void addInvoke(MachineBasicBlock *LandingPad,
                 MCSymbol *BeginLabel, MCSymbol *EndLabel);

  /// Add a new panding pad.  Returns the label ID for the landing pad entry.
  MCSymbol *addLandingPad(MachineBasicBlock *LandingPad);

  /// Provide the personality function for the exception information.
  void addPersonality(const Function *Personality);

  /// Return array of personality functions ever seen.
  const std::vector<const Function *>& getPersonalities() const {
    return Personalities;
  }

  /// Provide the catch typeinfo for a landing pad.
  void addCatchTypeInfo(MachineBasicBlock *LandingPad,
                        ArrayRef<const GlobalValue *> TyInfo);

  /// Provide the filter typeinfo for a landing pad.
  void addFilterTypeInfo(MachineBasicBlock *LandingPad,
                         ArrayRef<const GlobalValue *> TyInfo);

  /// Add a cleanup action for a landing pad.
  void addCleanup(MachineBasicBlock *LandingPad);

  void addSEHCatchHandler(MachineBasicBlock *LandingPad, const Function *Filter,
                          const BlockAddress *RecoverLabel);

  void addSEHCleanupHandler(MachineBasicBlock *LandingPad,
                            const Function *Cleanup);

  /// Return the type id for the specified typeinfo.  This is function wide.
  unsigned getTypeIDFor(const GlobalValue *TI);

  /// Return the id of the filter encoded by TyIds.  This is function wide.
  int getFilterIDFor(std::vector<unsigned> &TyIds);

  /// Remap landing pad labels and remove any deleted landing pads.
  void TidyLandingPads(DenseMap<MCSymbol*, uintptr_t> *LPMap = nullptr);

  /// Return a reference to the landing pad info for the current function.
  const std::vector<LandingPadInfo> &getLandingPads() const {
    return LandingPads;
  }

  /// Map the landing pad's EH symbol to the call site indexes.
  void setCallSiteLandingPad(MCSymbol *Sym, ArrayRef<unsigned> Sites);

  /// Get the call site indexes for a landing pad EH symbol.
  SmallVectorImpl<unsigned> &getCallSiteLandingPad(MCSymbol *Sym) {
    assert(hasCallSiteLandingPad(Sym) &&
           "missing call site number for landing pad!");
    return LPadToCallSiteMap[Sym];
  }

  /// Return true if the landing pad Eh symbol has an associated call site.
  bool hasCallSiteLandingPad(MCSymbol *Sym) {
    return !LPadToCallSiteMap[Sym].empty();
  }

  /// Map the begin label for a call site.
  void setCallSiteBeginLabel(MCSymbol *BeginLabel, unsigned Site) {
    CallSiteMap[BeginLabel] = Site;
  }

  /// Get the call site number for a begin label.
  unsigned getCallSiteBeginLabel(MCSymbol *BeginLabel) {
    assert(hasCallSiteBeginLabel(BeginLabel) &&
           "Missing call site number for EH_LABEL!");
    return CallSiteMap[BeginLabel];
  }

  /// Return true if the begin label has a call site number associated with it.
  bool hasCallSiteBeginLabel(MCSymbol *BeginLabel) {
    return CallSiteMap[BeginLabel] != 0;
  }

  /// Set the call site currently being processed.
  void setCurrentCallSite(unsigned Site) { CurCallSite = Site; }

  /// Get the call site currently being processed, if any.  return zero if
  /// none.
  unsigned getCurrentCallSite() { return CurCallSite; }

  /// Return a reference to the C++ typeinfo for the current function.
  const std::vector<const GlobalValue *> &getTypeInfos() const {
    return TypeInfos;
  }

  /// Return a reference to the typeids encoding filters used in the current
  /// function.
  const std::vector<unsigned> &getFilterIds() const {
    return FilterIds;
  }

  /// Collect information used to emit debugging information of a variable.
  void setVariableDbgInfo(const DILocalVariable *Var, const DIExpression *Expr,
                          unsigned Slot, const DILocation *Loc) {
    VariableDbgInfos.emplace_back(Var, Expr, Slot, Loc);
  }

  VariableDbgInfoMapTy &getVariableDbgInfo() { return VariableDbgInfos; }

}; // End class MachineModuleInfo

//===- MMI building helpers -----------------------------------------------===//

/// Determine if any floating-point values are being passed to this variadic
/// function, and set the MachineModuleInfo's usesVAFloatArgument flag if so.
/// This flag is used to emit an undefined reference to _fltused on Windows,
/// which will link in MSVCRT's floating-point support.
void computeUsesVAFloatArgument(const CallInst &I, MachineModuleInfo &MMI);

/// Extract the exception handling information from the landingpad instruction
/// and add them to the specified machine module info.
void addLandingPadInfo(const LandingPadInst &I, MachineModuleInfo &MMI,
                       MachineBasicBlock &MBB);


} // End llvm namespace

#endif
