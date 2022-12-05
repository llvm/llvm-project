//===- SPIRVModuleAnalysis.h - analysis of global instrs & regs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The analysis collects instructions that should be output at the module level
// and performs the global register numbering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVMODULEANALYSIS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVMODULEANALYSIS_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class SPIRVSubtarget;
class MachineFunction;
class MachineModuleInfo;

namespace SPIRV {
// The enum contains logical module sections for the instruction collection.
enum ModuleSectionType {
  //  MB_Capabilities, MB_Extensions, MB_ExtInstImports, MB_MemoryModel,
  MB_EntryPoints, // All OpEntryPoint instructions (if any).
  //  MB_ExecutionModes, MB_DebugSourceAndStrings,
  MB_DebugNames,           // All OpName and OpMemberName intrs.
  MB_DebugModuleProcessed, // All OpModuleProcessed instructions.
  MB_Annotations,          // OpDecorate, OpMemberDecorate etc.
  MB_TypeConstVars,        // OpTypeXXX, OpConstantXXX, and global OpVariables.
  MB_ExtFuncDecls,         // OpFunction etc. to declare for external funcs.
  NUM_MODULE_SECTIONS      // Total number of sections requiring basic blocks.
};

struct Requirements {
  const bool IsSatisfiable;
  const std::optional<Capability::Capability> Cap;
  const ExtensionList Exts;
  const unsigned MinVer; // 0 if no min version is required.
  const unsigned MaxVer; // 0 if no max version is required.

  Requirements(bool IsSatisfiable = false,
               std::optional<Capability::Capability> Cap = {},
               ExtensionList Exts = {}, unsigned MinVer = 0,
               unsigned MaxVer = 0)
      : IsSatisfiable(IsSatisfiable), Cap(Cap), Exts(Exts), MinVer(MinVer),
        MaxVer(MaxVer) {}
  Requirements(Capability::Capability Cap) : Requirements(true, {Cap}) {}
};

struct RequirementHandler {
private:
  CapabilityList MinimalCaps;
  SmallSet<Capability::Capability, 8> AllCaps;
  SmallSet<Extension::Extension, 4> AllExtensions;
  unsigned MinVersion; // 0 if no min version is defined.
  unsigned MaxVersion; // 0 if no max version is defined.
  DenseSet<unsigned> AvailableCaps;
  // Remove a list of capabilities from dedupedCaps and add them to AllCaps,
  // recursing through their implicitly declared capabilities too.
  void pruneCapabilities(const CapabilityList &ToPrune);

public:
  RequirementHandler() : MinVersion(0), MaxVersion(0) {}
  void clear() {
    MinimalCaps.clear();
    AllCaps.clear();
    AvailableCaps.clear();
    AllExtensions.clear();
    MinVersion = 0;
    MaxVersion = 0;
  }
  unsigned getMinVersion() const { return MinVersion; }
  unsigned getMaxVersion() const { return MaxVersion; }
  const CapabilityList &getMinimalCapabilities() const { return MinimalCaps; }
  const SmallSet<Extension::Extension, 4> &getExtensions() const {
    return AllExtensions;
  }
  // Add a list of capabilities, ensuring AllCaps captures all the implicitly
  // declared capabilities, and MinimalCaps has the minimal set of required
  // capabilities (so all implicitly declared ones are removed).
  void addCapabilities(const CapabilityList &ToAdd);
  void addCapability(Capability::Capability ToAdd) { addCapabilities({ToAdd}); }
  void addExtensions(const ExtensionList &ToAdd) {
    AllExtensions.insert(ToAdd.begin(), ToAdd.end());
  }
  void addExtension(Extension::Extension ToAdd) { AllExtensions.insert(ToAdd); }
  // Add the given requirements to the lists. If constraints conflict, or these
  // requirements cannot be satisfied, then abort the compilation.
  void addRequirements(const Requirements &Req);
  // Get requirement and add it to the list.
  void getAndAddRequirements(SPIRV::OperandCategory::OperandCategory Category,
                             uint32_t i, const SPIRVSubtarget &ST);
  // Check if all the requirements can be satisfied for the given subtarget, and
  // if not abort compilation.
  void checkSatisfiable(const SPIRVSubtarget &ST) const;
  void initAvailableCapabilities(const SPIRVSubtarget &ST);
  // Add the given capabilities to available and all their implicitly defined
  // capabilities too.
  void addAvailableCaps(const CapabilityList &ToAdd);
  bool isCapabilityAvailable(Capability::Capability Cap) const {
    return AvailableCaps.contains(Cap);
  }
};

using InstrList = SmallVector<MachineInstr *>;
// Maps a local register to the corresponding global alias.
using LocalToGlobalRegTable = std::map<Register, Register>;
using RegisterAliasMapTy =
    std::map<const MachineFunction *, LocalToGlobalRegTable>;

// The struct contains results of the module analysis and methods
// to access them.
struct ModuleAnalysisInfo {
  RequirementHandler Reqs;
  MemoryModel::MemoryModel Mem;
  AddressingModel::AddressingModel Addr;
  SourceLanguage::SourceLanguage SrcLang;
  unsigned SrcLangVersion;
  StringSet<> SrcExt;
  // Maps ExtInstSet to corresponding ID register.
  DenseMap<unsigned, Register> ExtInstSetMap;
  // Contains the list of all global OpVariables in the module.
  SmallVector<MachineInstr *, 4> GlobalVarList;
  // Maps function names to coresponding function ID registers.
  StringMap<Register> FuncNameMap;
  // The set contains machine instructions which are necessary
  // for correct MIR but will not be emitted in function bodies.
  DenseSet<MachineInstr *> InstrsToDelete;
  // The set contains machine basic blocks which are necessary
  // for correct MIR but will not be emitted.
  DenseSet<MachineBasicBlock *> MBBsToSkip;
  // The table contains global aliases of local registers for each machine
  // function. The aliases are used to substitute local registers during
  // code emission.
  RegisterAliasMapTy RegisterAliasTable;
  // The counter holds the maximum ID we have in the module.
  unsigned MaxID;
  // The array contains lists of MIs for each module section.
  InstrList MS[NUM_MODULE_SECTIONS];
  // The table maps MBB number to SPIR-V unique ID register.
  DenseMap<int, Register> BBNumToRegMap;

  Register getFuncReg(const Function *F) {
    assert(F && "Function is null");
    auto FuncReg = FuncNameMap.find(getFunctionGlobalIdentifier(F));
    assert(FuncReg != FuncNameMap.end() && "Cannot find function Id");
    return FuncReg->second;
  }
  Register getExtInstSetReg(unsigned SetNum) { return ExtInstSetMap[SetNum]; }
  InstrList &getMSInstrs(unsigned MSType) { return MS[MSType]; }
  void setSkipEmission(MachineInstr *MI) { InstrsToDelete.insert(MI); }
  bool getSkipEmission(const MachineInstr *MI) {
    return InstrsToDelete.contains(MI);
  }
  void setRegisterAlias(const MachineFunction *MF, Register Reg,
                        Register AliasReg) {
    RegisterAliasTable[MF][Reg] = AliasReg;
  }
  Register getRegisterAlias(const MachineFunction *MF, Register Reg) {
    auto RI = RegisterAliasTable[MF].find(Reg);
    if (RI == RegisterAliasTable[MF].end()) {
      return Register(0);
    }
    return RegisterAliasTable[MF][Reg];
  }
  bool hasRegisterAlias(const MachineFunction *MF, Register Reg) {
    return RegisterAliasTable.find(MF) != RegisterAliasTable.end() &&
           RegisterAliasTable[MF].find(Reg) != RegisterAliasTable[MF].end();
  }
  unsigned getNextID() { return MaxID++; }
  bool hasMBBRegister(const MachineBasicBlock &MBB) {
    return BBNumToRegMap.find(MBB.getNumber()) != BBNumToRegMap.end();
  }
  // Convert MBB's number to corresponding ID register.
  Register getOrCreateMBBRegister(const MachineBasicBlock &MBB) {
    auto f = BBNumToRegMap.find(MBB.getNumber());
    if (f != BBNumToRegMap.end())
      return f->second;
    Register NewReg = Register::index2VirtReg(getNextID());
    BBNumToRegMap[MBB.getNumber()] = NewReg;
    return NewReg;
  }
};
} // namespace SPIRV

struct SPIRVModuleAnalysis : public ModulePass {
  static char ID;

public:
  SPIRVModuleAnalysis() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static struct SPIRV::ModuleAnalysisInfo MAI;

private:
  void setBaseInfo(const Module &M);
  void collectGlobalEntities(
      const std::vector<SPIRV::DTSortableEntry *> &DepsGraph,
      SPIRV::ModuleSectionType MSType,
      std::function<bool(const SPIRV::DTSortableEntry *)> Pred,
      bool UsePreOrder);
  void processDefInstrs(const Module &M);
  void collectFuncNames(MachineInstr &MI, const Function &F);
  void processOtherInstrs(const Module &M);
  void numberRegistersGlobally(const Module &M);

  const SPIRVSubtarget *ST;
  SPIRVGlobalRegistry *GR;
  const SPIRVInstrInfo *TII;
  MachineModuleInfo *MMI;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVMODULEANALYSIS_H
