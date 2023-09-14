//===- SPIRVModuleAnalysis.cpp - analysis of global instrs & regs - C++ -*-===//
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
// The results of this analysis are used in AsmPrinter to rename registers
// globally and to output required instructions at the module level.
//
//===----------------------------------------------------------------------===//

#include "SPIRVModuleAnalysis.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "TargetInfo/SPIRVTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-module-analysis"

static cl::opt<bool>
    SPVDumpDeps("spv-dump-deps",
                cl::desc("Dump MIR with SPIR-V dependencies info"),
                cl::Optional, cl::init(false));

char llvm::SPIRVModuleAnalysis::ID = 0;

namespace llvm {
void initializeSPIRVModuleAnalysisPass(PassRegistry &);
} // namespace llvm

INITIALIZE_PASS(SPIRVModuleAnalysis, DEBUG_TYPE, "SPIRV module analysis", true,
                true)

// Retrieve an unsigned from an MDNode with a list of them as operands.
static unsigned getMetadataUInt(MDNode *MdNode, unsigned OpIndex,
                                unsigned DefaultVal = 0) {
  if (MdNode && OpIndex < MdNode->getNumOperands()) {
    const auto &Op = MdNode->getOperand(OpIndex);
    return mdconst::extract<ConstantInt>(Op)->getZExtValue();
  }
  return DefaultVal;
}

static SPIRV::Requirements
getSymbolicOperandRequirements(SPIRV::OperandCategory::OperandCategory Category,
                               unsigned i, const SPIRVSubtarget &ST,
                               SPIRV::RequirementHandler &Reqs) {
  unsigned ReqMinVer = getSymbolicOperandMinVersion(Category, i);
  unsigned ReqMaxVer = getSymbolicOperandMaxVersion(Category, i);
  unsigned TargetVer = ST.getSPIRVVersion();
  bool MinVerOK = !ReqMinVer || !TargetVer || TargetVer >= ReqMinVer;
  bool MaxVerOK = !ReqMaxVer || !TargetVer || TargetVer <= ReqMaxVer;
  CapabilityList ReqCaps = getSymbolicOperandCapabilities(Category, i);
  ExtensionList ReqExts = getSymbolicOperandExtensions(Category, i);
  if (ReqCaps.empty()) {
    if (ReqExts.empty()) {
      if (MinVerOK && MaxVerOK)
        return {true, {}, {}, ReqMinVer, ReqMaxVer};
      return {false, {}, {}, 0, 0};
    }
  } else if (MinVerOK && MaxVerOK) {
    for (auto Cap : ReqCaps) { // Only need 1 of the capabilities to work.
      if (Reqs.isCapabilityAvailable(Cap))
        return {true, {Cap}, {}, ReqMinVer, ReqMaxVer};
    }
  }
  // If there are no capabilities, or we can't satisfy the version or
  // capability requirements, use the list of extensions (if the subtarget
  // can handle them all).
  if (llvm::all_of(ReqExts, [&ST](const SPIRV::Extension::Extension &Ext) {
        return ST.canUseExtension(Ext);
      })) {
    return {true, {}, ReqExts, 0, 0}; // TODO: add versions to extensions.
  }
  return {false, {}, {}, 0, 0};
}

void SPIRVModuleAnalysis::setBaseInfo(const Module &M) {
  MAI.MaxID = 0;
  for (int i = 0; i < SPIRV::NUM_MODULE_SECTIONS; i++)
    MAI.MS[i].clear();
  MAI.RegisterAliasTable.clear();
  MAI.InstrsToDelete.clear();
  MAI.FuncMap.clear();
  MAI.GlobalVarList.clear();
  MAI.ExtInstSetMap.clear();
  MAI.Reqs.clear();
  MAI.Reqs.initAvailableCapabilities(*ST);

  // TODO: determine memory model and source language from the configuratoin.
  if (auto MemModel = M.getNamedMetadata("spirv.MemoryModel")) {
    auto MemMD = MemModel->getOperand(0);
    MAI.Addr = static_cast<SPIRV::AddressingModel::AddressingModel>(
        getMetadataUInt(MemMD, 0));
    MAI.Mem =
        static_cast<SPIRV::MemoryModel::MemoryModel>(getMetadataUInt(MemMD, 1));
  } else {
    // TODO: Add support for VulkanMemoryModel.
    MAI.Mem = ST->isOpenCLEnv() ? SPIRV::MemoryModel::OpenCL
                                : SPIRV::MemoryModel::GLSL450;
    if (MAI.Mem == SPIRV::MemoryModel::OpenCL) {
      unsigned PtrSize = ST->getPointerSize();
      MAI.Addr = PtrSize == 32   ? SPIRV::AddressingModel::Physical32
                 : PtrSize == 64 ? SPIRV::AddressingModel::Physical64
                                 : SPIRV::AddressingModel::Logical;
    } else {
      // TODO: Add support for PhysicalStorageBufferAddress.
      MAI.Addr = SPIRV::AddressingModel::Logical;
    }
  }
  // Get the OpenCL version number from metadata.
  // TODO: support other source languages.
  if (auto VerNode = M.getNamedMetadata("opencl.ocl.version")) {
    MAI.SrcLang = SPIRV::SourceLanguage::OpenCL_C;
    // Construct version literal in accordance with SPIRV-LLVM-Translator.
    // TODO: support multiple OCL version metadata.
    assert(VerNode->getNumOperands() > 0 && "Invalid SPIR");
    auto VersionMD = VerNode->getOperand(0);
    unsigned MajorNum = getMetadataUInt(VersionMD, 0, 2);
    unsigned MinorNum = getMetadataUInt(VersionMD, 1);
    unsigned RevNum = getMetadataUInt(VersionMD, 2);
    MAI.SrcLangVersion = (MajorNum * 100 + MinorNum) * 1000 + RevNum;
  } else {
    MAI.SrcLang = SPIRV::SourceLanguage::Unknown;
    MAI.SrcLangVersion = 0;
  }

  if (auto ExtNode = M.getNamedMetadata("opencl.used.extensions")) {
    for (unsigned I = 0, E = ExtNode->getNumOperands(); I != E; ++I) {
      MDNode *MD = ExtNode->getOperand(I);
      if (!MD || MD->getNumOperands() == 0)
        continue;
      for (unsigned J = 0, N = MD->getNumOperands(); J != N; ++J)
        MAI.SrcExt.insert(cast<MDString>(MD->getOperand(J))->getString());
    }
  }

  // Update required capabilities for this memory model, addressing model and
  // source language.
  MAI.Reqs.getAndAddRequirements(SPIRV::OperandCategory::MemoryModelOperand,
                                 MAI.Mem, *ST);
  MAI.Reqs.getAndAddRequirements(SPIRV::OperandCategory::SourceLanguageOperand,
                                 MAI.SrcLang, *ST);
  MAI.Reqs.getAndAddRequirements(SPIRV::OperandCategory::AddressingModelOperand,
                                 MAI.Addr, *ST);

  if (ST->isOpenCLEnv()) {
    // TODO: check if it's required by default.
    MAI.ExtInstSetMap[static_cast<unsigned>(
        SPIRV::InstructionSet::OpenCL_std)] =
        Register::index2VirtReg(MAI.getNextID());
  }
}

// Collect MI which defines the register in the given machine function.
static void collectDefInstr(Register Reg, const MachineFunction *MF,
                            SPIRV::ModuleAnalysisInfo *MAI,
                            SPIRV::ModuleSectionType MSType,
                            bool DoInsert = true) {
  assert(MAI->hasRegisterAlias(MF, Reg) && "Cannot find register alias");
  MachineInstr *MI = MF->getRegInfo().getUniqueVRegDef(Reg);
  assert(MI && "There should be an instruction that defines the register");
  MAI->setSkipEmission(MI);
  if (DoInsert)
    MAI->MS[MSType].push_back(MI);
}

void SPIRVModuleAnalysis::collectGlobalEntities(
    const std::vector<SPIRV::DTSortableEntry *> &DepsGraph,
    SPIRV::ModuleSectionType MSType,
    std::function<bool(const SPIRV::DTSortableEntry *)> Pred,
    bool UsePreOrder = false) {
  DenseSet<const SPIRV::DTSortableEntry *> Visited;
  for (const auto *E : DepsGraph) {
    std::function<void(const SPIRV::DTSortableEntry *)> RecHoistUtil;
    // NOTE: here we prefer recursive approach over iterative because
    // we don't expect depchains long enough to cause SO.
    RecHoistUtil = [MSType, UsePreOrder, &Visited, &Pred,
                    &RecHoistUtil](const SPIRV::DTSortableEntry *E) {
      if (Visited.count(E) || !Pred(E))
        return;
      Visited.insert(E);

      // Traversing deps graph in post-order allows us to get rid of
      // register aliases preprocessing.
      // But pre-order is required for correct processing of function
      // declaration and arguments processing.
      if (!UsePreOrder)
        for (auto *S : E->getDeps())
          RecHoistUtil(S);

      Register GlobalReg = Register::index2VirtReg(MAI.getNextID());
      bool IsFirst = true;
      for (auto &U : *E) {
        const MachineFunction *MF = U.first;
        Register Reg = U.second;
        MAI.setRegisterAlias(MF, Reg, GlobalReg);
        if (!MF->getRegInfo().getUniqueVRegDef(Reg))
          continue;
        collectDefInstr(Reg, MF, &MAI, MSType, IsFirst);
        IsFirst = false;
        if (E->getIsGV())
          MAI.GlobalVarList.push_back(MF->getRegInfo().getUniqueVRegDef(Reg));
      }

      if (UsePreOrder)
        for (auto *S : E->getDeps())
          RecHoistUtil(S);
    };
    RecHoistUtil(E);
  }
}

// The function initializes global register alias table for types, consts,
// global vars and func decls and collects these instruction for output
// at module level. Also it collects explicit OpExtension/OpCapability
// instructions.
void SPIRVModuleAnalysis::processDefInstrs(const Module &M) {
  std::vector<SPIRV::DTSortableEntry *> DepsGraph;

  GR->buildDepsGraph(DepsGraph, SPVDumpDeps ? MMI : nullptr);

  collectGlobalEntities(
      DepsGraph, SPIRV::MB_TypeConstVars,
      [](const SPIRV::DTSortableEntry *E) { return !E->getIsFunc(); });

  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    // Iterate through and collect OpExtension/OpCapability instructions.
    for (MachineBasicBlock &MBB : *MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.getOpcode() == SPIRV::OpExtension) {
          // Here, OpExtension just has a single enum operand, not a string.
          auto Ext = SPIRV::Extension::Extension(MI.getOperand(0).getImm());
          MAI.Reqs.addExtension(Ext);
          MAI.setSkipEmission(&MI);
        } else if (MI.getOpcode() == SPIRV::OpCapability) {
          auto Cap = SPIRV::Capability::Capability(MI.getOperand(0).getImm());
          MAI.Reqs.addCapability(Cap);
          MAI.setSkipEmission(&MI);
        }
      }
    }
  }

  collectGlobalEntities(
      DepsGraph, SPIRV::MB_ExtFuncDecls,
      [](const SPIRV::DTSortableEntry *E) { return E->getIsFunc(); }, true);
}

// True if there is an instruction in the MS list with all the same operands as
// the given instruction has (after the given starting index).
// TODO: maybe it needs to check Opcodes too.
static bool findSameInstrInMS(const MachineInstr &A,
                              SPIRV::ModuleSectionType MSType,
                              SPIRV::ModuleAnalysisInfo &MAI,
                              unsigned StartOpIndex = 0) {
  for (const auto *B : MAI.MS[MSType]) {
    const unsigned NumAOps = A.getNumOperands();
    if (NumAOps != B->getNumOperands() || A.getNumDefs() != B->getNumDefs())
      continue;
    bool AllOpsMatch = true;
    for (unsigned i = StartOpIndex; i < NumAOps && AllOpsMatch; ++i) {
      if (A.getOperand(i).isReg() && B->getOperand(i).isReg()) {
        Register RegA = A.getOperand(i).getReg();
        Register RegB = B->getOperand(i).getReg();
        AllOpsMatch = MAI.getRegisterAlias(A.getMF(), RegA) ==
                      MAI.getRegisterAlias(B->getMF(), RegB);
      } else {
        AllOpsMatch = A.getOperand(i).isIdenticalTo(B->getOperand(i));
      }
    }
    if (AllOpsMatch)
      return true;
  }
  return false;
}

// Look for IDs declared with Import linkage, and map the corresponding function
// to the register defining that variable (which will usually be the result of
// an OpFunction). This lets us call externally imported functions using
// the correct ID registers.
void SPIRVModuleAnalysis::collectFuncNames(MachineInstr &MI,
                                           const Function *F) {
  if (MI.getOpcode() == SPIRV::OpDecorate) {
    // If it's got Import linkage.
    auto Dec = MI.getOperand(1).getImm();
    if (Dec == static_cast<unsigned>(SPIRV::Decoration::LinkageAttributes)) {
      auto Lnk = MI.getOperand(MI.getNumOperands() - 1).getImm();
      if (Lnk == static_cast<unsigned>(SPIRV::LinkageType::Import)) {
        // Map imported function name to function ID register.
        const Function *ImportedFunc =
            F->getParent()->getFunction(getStringImm(MI, 2));
        Register Target = MI.getOperand(0).getReg();
        MAI.FuncMap[ImportedFunc] = MAI.getRegisterAlias(MI.getMF(), Target);
      }
    }
  } else if (MI.getOpcode() == SPIRV::OpFunction) {
    // Record all internal OpFunction declarations.
    Register Reg = MI.defs().begin()->getReg();
    Register GlobalReg = MAI.getRegisterAlias(MI.getMF(), Reg);
    assert(GlobalReg.isValid());
    MAI.FuncMap[F] = GlobalReg;
  }
}

// Collect the given instruction in the specified MS. We assume global register
// numbering has already occurred by this point. We can directly compare reg
// arguments when detecting duplicates.
static void collectOtherInstr(MachineInstr &MI, SPIRV::ModuleAnalysisInfo &MAI,
                              SPIRV::ModuleSectionType MSType,
                              bool Append = true) {
  MAI.setSkipEmission(&MI);
  if (findSameInstrInMS(MI, MSType, MAI))
    return; // Found a duplicate, so don't add it.
  // No duplicates, so add it.
  if (Append)
    MAI.MS[MSType].push_back(&MI);
  else
    MAI.MS[MSType].insert(MAI.MS[MSType].begin(), &MI);
}

// Some global instructions make reference to function-local ID regs, so cannot
// be correctly collected until these registers are globally numbered.
void SPIRVModuleAnalysis::processOtherInstrs(const Module &M) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    if ((*F).isDeclaration())
      continue;
    MachineFunction *MF = MMI->getMachineFunction(*F);
    assert(MF);
    for (MachineBasicBlock &MBB : *MF)
      for (MachineInstr &MI : MBB) {
        if (MAI.getSkipEmission(&MI))
          continue;
        const unsigned OpCode = MI.getOpcode();
        if (OpCode == SPIRV::OpName || OpCode == SPIRV::OpMemberName) {
          collectOtherInstr(MI, MAI, SPIRV::MB_DebugNames);
        } else if (OpCode == SPIRV::OpEntryPoint) {
          collectOtherInstr(MI, MAI, SPIRV::MB_EntryPoints);
        } else if (TII->isDecorationInstr(MI)) {
          collectOtherInstr(MI, MAI, SPIRV::MB_Annotations);
          collectFuncNames(MI, &*F);
        } else if (TII->isConstantInstr(MI)) {
          // Now OpSpecConstant*s are not in DT,
          // but they need to be collected anyway.
          collectOtherInstr(MI, MAI, SPIRV::MB_TypeConstVars);
        } else if (OpCode == SPIRV::OpFunction) {
          collectFuncNames(MI, &*F);
        } else if (OpCode == SPIRV::OpTypeForwardPointer) {
          collectOtherInstr(MI, MAI, SPIRV::MB_TypeConstVars, false);
        }
      }
  }
}

// Number registers in all functions globally from 0 onwards and store
// the result in global register alias table. Some registers are already
// numbered in collectGlobalEntities.
void SPIRVModuleAnalysis::numberRegistersGlobally(const Module &M) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    if ((*F).isDeclaration())
      continue;
    MachineFunction *MF = MMI->getMachineFunction(*F);
    assert(MF);
    for (MachineBasicBlock &MBB : *MF) {
      for (MachineInstr &MI : MBB) {
        for (MachineOperand &Op : MI.operands()) {
          if (!Op.isReg())
            continue;
          Register Reg = Op.getReg();
          if (MAI.hasRegisterAlias(MF, Reg))
            continue;
          Register NewReg = Register::index2VirtReg(MAI.getNextID());
          MAI.setRegisterAlias(MF, Reg, NewReg);
        }
        if (MI.getOpcode() != SPIRV::OpExtInst)
          continue;
        auto Set = MI.getOperand(2).getImm();
        if (MAI.ExtInstSetMap.find(Set) == MAI.ExtInstSetMap.end())
          MAI.ExtInstSetMap[Set] = Register::index2VirtReg(MAI.getNextID());
      }
    }
  }
}

// RequirementHandler implementations.
void SPIRV::RequirementHandler::getAndAddRequirements(
    SPIRV::OperandCategory::OperandCategory Category, uint32_t i,
    const SPIRVSubtarget &ST) {
  addRequirements(getSymbolicOperandRequirements(Category, i, ST, *this));
}

void SPIRV::RequirementHandler::pruneCapabilities(
    const CapabilityList &ToPrune) {
  for (const auto &Cap : ToPrune) {
    AllCaps.insert(Cap);
    auto FoundIndex = std::find(MinimalCaps.begin(), MinimalCaps.end(), Cap);
    if (FoundIndex != MinimalCaps.end())
      MinimalCaps.erase(FoundIndex);
    CapabilityList ImplicitDecls =
        getSymbolicOperandCapabilities(OperandCategory::CapabilityOperand, Cap);
    pruneCapabilities(ImplicitDecls);
  }
}

void SPIRV::RequirementHandler::addCapabilities(const CapabilityList &ToAdd) {
  for (const auto &Cap : ToAdd) {
    bool IsNewlyInserted = AllCaps.insert(Cap).second;
    if (!IsNewlyInserted) // Don't re-add if it's already been declared.
      continue;
    CapabilityList ImplicitDecls =
        getSymbolicOperandCapabilities(OperandCategory::CapabilityOperand, Cap);
    pruneCapabilities(ImplicitDecls);
    MinimalCaps.push_back(Cap);
  }
}

void SPIRV::RequirementHandler::addRequirements(
    const SPIRV::Requirements &Req) {
  if (!Req.IsSatisfiable)
    report_fatal_error("Adding SPIR-V requirements this target can't satisfy.");

  if (Req.Cap.has_value())
    addCapabilities({Req.Cap.value()});

  addExtensions(Req.Exts);

  if (Req.MinVer) {
    if (MaxVersion && Req.MinVer > MaxVersion) {
      LLVM_DEBUG(dbgs() << "Conflicting version requirements: >= " << Req.MinVer
                        << " and <= " << MaxVersion << "\n");
      report_fatal_error("Adding SPIR-V requirements that can't be satisfied.");
    }

    if (MinVersion == 0 || Req.MinVer > MinVersion)
      MinVersion = Req.MinVer;
  }

  if (Req.MaxVer) {
    if (MinVersion && Req.MaxVer < MinVersion) {
      LLVM_DEBUG(dbgs() << "Conflicting version requirements: <= " << Req.MaxVer
                        << " and >= " << MinVersion << "\n");
      report_fatal_error("Adding SPIR-V requirements that can't be satisfied.");
    }

    if (MaxVersion == 0 || Req.MaxVer < MaxVersion)
      MaxVersion = Req.MaxVer;
  }
}

void SPIRV::RequirementHandler::checkSatisfiable(
    const SPIRVSubtarget &ST) const {
  // Report as many errors as possible before aborting the compilation.
  bool IsSatisfiable = true;
  auto TargetVer = ST.getSPIRVVersion();

  if (MaxVersion && TargetVer && MaxVersion < TargetVer) {
    LLVM_DEBUG(
        dbgs() << "Target SPIR-V version too high for required features\n"
               << "Required max version: " << MaxVersion << " target version "
               << TargetVer << "\n");
    IsSatisfiable = false;
  }

  if (MinVersion && TargetVer && MinVersion > TargetVer) {
    LLVM_DEBUG(dbgs() << "Target SPIR-V version too low for required features\n"
                      << "Required min version: " << MinVersion
                      << " target version " << TargetVer << "\n");
    IsSatisfiable = false;
  }

  if (MinVersion && MaxVersion && MinVersion > MaxVersion) {
    LLVM_DEBUG(
        dbgs()
        << "Version is too low for some features and too high for others.\n"
        << "Required SPIR-V min version: " << MinVersion
        << " required SPIR-V max version " << MaxVersion << "\n");
    IsSatisfiable = false;
  }

  for (auto Cap : MinimalCaps) {
    if (AvailableCaps.contains(Cap))
      continue;
    LLVM_DEBUG(dbgs() << "Capability not supported: "
                      << getSymbolicOperandMnemonic(
                             OperandCategory::CapabilityOperand, Cap)
                      << "\n");
    IsSatisfiable = false;
  }

  for (auto Ext : AllExtensions) {
    if (ST.canUseExtension(Ext))
      continue;
    LLVM_DEBUG(dbgs() << "Extension not suported: "
                      << getSymbolicOperandMnemonic(
                             OperandCategory::ExtensionOperand, Ext)
                      << "\n");
    IsSatisfiable = false;
  }

  if (!IsSatisfiable)
    report_fatal_error("Unable to meet SPIR-V requirements for this target.");
}

// Add the given capabilities and all their implicitly defined capabilities too.
void SPIRV::RequirementHandler::addAvailableCaps(const CapabilityList &ToAdd) {
  for (const auto Cap : ToAdd)
    if (AvailableCaps.insert(Cap).second)
      addAvailableCaps(getSymbolicOperandCapabilities(
          SPIRV::OperandCategory::CapabilityOperand, Cap));
}

namespace llvm {
namespace SPIRV {
void RequirementHandler::initAvailableCapabilities(const SPIRVSubtarget &ST) {
  if (ST.isOpenCLEnv()) {
    initAvailableCapabilitiesForOpenCL(ST);
    return;
  }

  if (ST.isVulkanEnv()) {
    initAvailableCapabilitiesForVulkan(ST);
    return;
  }

  report_fatal_error("Unimplemented environment for SPIR-V generation.");
}

void RequirementHandler::initAvailableCapabilitiesForOpenCL(
    const SPIRVSubtarget &ST) {
  // Add the min requirements for different OpenCL and SPIR-V versions.
  addAvailableCaps({Capability::Addresses, Capability::Float16Buffer,
                    Capability::Int16, Capability::Int8, Capability::Kernel,
                    Capability::Linkage, Capability::Vector16,
                    Capability::Groups, Capability::GenericPointer,
                    Capability::Shader});
  if (ST.hasOpenCLFullProfile())
    addAvailableCaps({Capability::Int64, Capability::Int64Atomics});
  if (ST.hasOpenCLImageSupport()) {
    addAvailableCaps({Capability::ImageBasic, Capability::LiteralSampler,
                      Capability::Image1D, Capability::SampledBuffer,
                      Capability::ImageBuffer});
    if (ST.isAtLeastOpenCLVer(20))
      addAvailableCaps({Capability::ImageReadWrite});
  }
  if (ST.isAtLeastSPIRVVer(11) && ST.isAtLeastOpenCLVer(22))
    addAvailableCaps({Capability::SubgroupDispatch, Capability::PipeStorage});
  if (ST.isAtLeastSPIRVVer(13))
    addAvailableCaps({Capability::GroupNonUniform,
                      Capability::GroupNonUniformVote,
                      Capability::GroupNonUniformArithmetic,
                      Capability::GroupNonUniformBallot,
                      Capability::GroupNonUniformClustered,
                      Capability::GroupNonUniformShuffle,
                      Capability::GroupNonUniformShuffleRelative});
  if (ST.isAtLeastSPIRVVer(14))
    addAvailableCaps({Capability::DenormPreserve, Capability::DenormFlushToZero,
                      Capability::SignedZeroInfNanPreserve,
                      Capability::RoundingModeRTE,
                      Capability::RoundingModeRTZ});
  // TODO: verify if this needs some checks.
  addAvailableCaps({Capability::Float16, Capability::Float64});

  // Add capabilities enabled by extensions.
  for (auto Extension : ST.getAllAvailableExtensions()) {
    CapabilityList EnabledCapabilities =
        getCapabilitiesEnabledByExtension(Extension);
    addAvailableCaps(EnabledCapabilities);
  }

  // TODO: add OpenCL extensions.
}

void RequirementHandler::initAvailableCapabilitiesForVulkan(
    const SPIRVSubtarget &ST) {
  addAvailableCaps({Capability::Shader, Capability::Linkage});
}

} // namespace SPIRV
} // namespace llvm

// Add the required capabilities from a decoration instruction (including
// BuiltIns).
static void addOpDecorateReqs(const MachineInstr &MI, unsigned DecIndex,
                              SPIRV::RequirementHandler &Reqs,
                              const SPIRVSubtarget &ST) {
  int64_t DecOp = MI.getOperand(DecIndex).getImm();
  auto Dec = static_cast<SPIRV::Decoration::Decoration>(DecOp);
  Reqs.addRequirements(getSymbolicOperandRequirements(
      SPIRV::OperandCategory::DecorationOperand, Dec, ST, Reqs));

  if (Dec == SPIRV::Decoration::BuiltIn) {
    int64_t BuiltInOp = MI.getOperand(DecIndex + 1).getImm();
    auto BuiltIn = static_cast<SPIRV::BuiltIn::BuiltIn>(BuiltInOp);
    Reqs.addRequirements(getSymbolicOperandRequirements(
        SPIRV::OperandCategory::BuiltInOperand, BuiltIn, ST, Reqs));
  }
}

// Add requirements for image handling.
static void addOpTypeImageReqs(const MachineInstr &MI,
                               SPIRV::RequirementHandler &Reqs,
                               const SPIRVSubtarget &ST) {
  assert(MI.getNumOperands() >= 8 && "Insufficient operands for OpTypeImage");
  // The operand indices used here are based on the OpTypeImage layout, which
  // the MachineInstr follows as well.
  int64_t ImgFormatOp = MI.getOperand(7).getImm();
  auto ImgFormat = static_cast<SPIRV::ImageFormat::ImageFormat>(ImgFormatOp);
  Reqs.getAndAddRequirements(SPIRV::OperandCategory::ImageFormatOperand,
                             ImgFormat, ST);

  bool IsArrayed = MI.getOperand(4).getImm() == 1;
  bool IsMultisampled = MI.getOperand(5).getImm() == 1;
  bool NoSampler = MI.getOperand(6).getImm() == 2;
  // Add dimension requirements.
  assert(MI.getOperand(2).isImm());
  switch (MI.getOperand(2).getImm()) {
  case SPIRV::Dim::DIM_1D:
    Reqs.addRequirements(NoSampler ? SPIRV::Capability::Image1D
                                   : SPIRV::Capability::Sampled1D);
    break;
  case SPIRV::Dim::DIM_2D:
    if (IsMultisampled && NoSampler)
      Reqs.addRequirements(SPIRV::Capability::ImageMSArray);
    break;
  case SPIRV::Dim::DIM_Cube:
    Reqs.addRequirements(SPIRV::Capability::Shader);
    if (IsArrayed)
      Reqs.addRequirements(NoSampler ? SPIRV::Capability::ImageCubeArray
                                     : SPIRV::Capability::SampledCubeArray);
    break;
  case SPIRV::Dim::DIM_Rect:
    Reqs.addRequirements(NoSampler ? SPIRV::Capability::ImageRect
                                   : SPIRV::Capability::SampledRect);
    break;
  case SPIRV::Dim::DIM_Buffer:
    Reqs.addRequirements(NoSampler ? SPIRV::Capability::ImageBuffer
                                   : SPIRV::Capability::SampledBuffer);
    break;
  case SPIRV::Dim::DIM_SubpassData:
    Reqs.addRequirements(SPIRV::Capability::InputAttachment);
    break;
  }

  // Has optional access qualifier.
  // TODO: check if it's OpenCL's kernel.
  if (MI.getNumOperands() > 8 &&
      MI.getOperand(8).getImm() == SPIRV::AccessQualifier::ReadWrite)
    Reqs.addRequirements(SPIRV::Capability::ImageReadWrite);
  else
    Reqs.addRequirements(SPIRV::Capability::ImageBasic);
}

void addInstrRequirements(const MachineInstr &MI,
                          SPIRV::RequirementHandler &Reqs,
                          const SPIRVSubtarget &ST) {
  switch (MI.getOpcode()) {
  case SPIRV::OpMemoryModel: {
    int64_t Addr = MI.getOperand(0).getImm();
    Reqs.getAndAddRequirements(SPIRV::OperandCategory::AddressingModelOperand,
                               Addr, ST);
    int64_t Mem = MI.getOperand(1).getImm();
    Reqs.getAndAddRequirements(SPIRV::OperandCategory::MemoryModelOperand, Mem,
                               ST);
    break;
  }
  case SPIRV::OpEntryPoint: {
    int64_t Exe = MI.getOperand(0).getImm();
    Reqs.getAndAddRequirements(SPIRV::OperandCategory::ExecutionModelOperand,
                               Exe, ST);
    break;
  }
  case SPIRV::OpExecutionMode:
  case SPIRV::OpExecutionModeId: {
    int64_t Exe = MI.getOperand(1).getImm();
    Reqs.getAndAddRequirements(SPIRV::OperandCategory::ExecutionModeOperand,
                               Exe, ST);
    break;
  }
  case SPIRV::OpTypeMatrix:
    Reqs.addCapability(SPIRV::Capability::Matrix);
    break;
  case SPIRV::OpTypeInt: {
    unsigned BitWidth = MI.getOperand(1).getImm();
    if (BitWidth == 64)
      Reqs.addCapability(SPIRV::Capability::Int64);
    else if (BitWidth == 16)
      Reqs.addCapability(SPIRV::Capability::Int16);
    else if (BitWidth == 8)
      Reqs.addCapability(SPIRV::Capability::Int8);
    break;
  }
  case SPIRV::OpTypeFloat: {
    unsigned BitWidth = MI.getOperand(1).getImm();
    if (BitWidth == 64)
      Reqs.addCapability(SPIRV::Capability::Float64);
    else if (BitWidth == 16)
      Reqs.addCapability(SPIRV::Capability::Float16);
    break;
  }
  case SPIRV::OpTypeVector: {
    unsigned NumComponents = MI.getOperand(2).getImm();
    if (NumComponents == 8 || NumComponents == 16)
      Reqs.addCapability(SPIRV::Capability::Vector16);
    break;
  }
  case SPIRV::OpTypePointer: {
    auto SC = MI.getOperand(1).getImm();
    Reqs.getAndAddRequirements(SPIRV::OperandCategory::StorageClassOperand, SC,
                               ST);
    // If it's a type of pointer to float16, add Float16Buffer capability.
    assert(MI.getOperand(2).isReg());
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    SPIRVType *TypeDef = MRI.getVRegDef(MI.getOperand(2).getReg());
    if (TypeDef->getOpcode() == SPIRV::OpTypeFloat &&
        TypeDef->getOperand(1).getImm() == 16)
      Reqs.addCapability(SPIRV::Capability::Float16Buffer);
    break;
  }
  case SPIRV::OpBitReverse:
  case SPIRV::OpTypeRuntimeArray:
    Reqs.addCapability(SPIRV::Capability::Shader);
    break;
  case SPIRV::OpTypeOpaque:
  case SPIRV::OpTypeEvent:
    Reqs.addCapability(SPIRV::Capability::Kernel);
    break;
  case SPIRV::OpTypePipe:
  case SPIRV::OpTypeReserveId:
    Reqs.addCapability(SPIRV::Capability::Pipes);
    break;
  case SPIRV::OpTypeDeviceEvent:
  case SPIRV::OpTypeQueue:
  case SPIRV::OpBuildNDRange:
    Reqs.addCapability(SPIRV::Capability::DeviceEnqueue);
    break;
  case SPIRV::OpDecorate:
  case SPIRV::OpDecorateId:
  case SPIRV::OpDecorateString:
    addOpDecorateReqs(MI, 1, Reqs, ST);
    break;
  case SPIRV::OpMemberDecorate:
  case SPIRV::OpMemberDecorateString:
    addOpDecorateReqs(MI, 2, Reqs, ST);
    break;
  case SPIRV::OpInBoundsPtrAccessChain:
    Reqs.addCapability(SPIRV::Capability::Addresses);
    break;
  case SPIRV::OpConstantSampler:
    Reqs.addCapability(SPIRV::Capability::LiteralSampler);
    break;
  case SPIRV::OpTypeImage:
    addOpTypeImageReqs(MI, Reqs, ST);
    break;
  case SPIRV::OpTypeSampler:
    Reqs.addCapability(SPIRV::Capability::ImageBasic);
    break;
  case SPIRV::OpTypeForwardPointer:
    // TODO: check if it's OpenCL's kernel.
    Reqs.addCapability(SPIRV::Capability::Addresses);
    break;
  case SPIRV::OpAtomicFlagTestAndSet:
  case SPIRV::OpAtomicLoad:
  case SPIRV::OpAtomicStore:
  case SPIRV::OpAtomicExchange:
  case SPIRV::OpAtomicCompareExchange:
  case SPIRV::OpAtomicIIncrement:
  case SPIRV::OpAtomicIDecrement:
  case SPIRV::OpAtomicIAdd:
  case SPIRV::OpAtomicISub:
  case SPIRV::OpAtomicUMin:
  case SPIRV::OpAtomicUMax:
  case SPIRV::OpAtomicSMin:
  case SPIRV::OpAtomicSMax:
  case SPIRV::OpAtomicAnd:
  case SPIRV::OpAtomicOr:
  case SPIRV::OpAtomicXor: {
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    const MachineInstr *InstrPtr = &MI;
    if (MI.getOpcode() == SPIRV::OpAtomicStore) {
      assert(MI.getOperand(3).isReg());
      InstrPtr = MRI.getVRegDef(MI.getOperand(3).getReg());
      assert(InstrPtr && "Unexpected type instruction for OpAtomicStore");
    }
    assert(InstrPtr->getOperand(1).isReg() && "Unexpected operand in atomic");
    Register TypeReg = InstrPtr->getOperand(1).getReg();
    SPIRVType *TypeDef = MRI.getVRegDef(TypeReg);
    if (TypeDef->getOpcode() == SPIRV::OpTypeInt) {
      unsigned BitWidth = TypeDef->getOperand(1).getImm();
      if (BitWidth == 64)
        Reqs.addCapability(SPIRV::Capability::Int64Atomics);
    }
    break;
  }
  case SPIRV::OpGroupNonUniformIAdd:
  case SPIRV::OpGroupNonUniformFAdd:
  case SPIRV::OpGroupNonUniformIMul:
  case SPIRV::OpGroupNonUniformFMul:
  case SPIRV::OpGroupNonUniformSMin:
  case SPIRV::OpGroupNonUniformUMin:
  case SPIRV::OpGroupNonUniformFMin:
  case SPIRV::OpGroupNonUniformSMax:
  case SPIRV::OpGroupNonUniformUMax:
  case SPIRV::OpGroupNonUniformFMax:
  case SPIRV::OpGroupNonUniformBitwiseAnd:
  case SPIRV::OpGroupNonUniformBitwiseOr:
  case SPIRV::OpGroupNonUniformBitwiseXor:
  case SPIRV::OpGroupNonUniformLogicalAnd:
  case SPIRV::OpGroupNonUniformLogicalOr:
  case SPIRV::OpGroupNonUniformLogicalXor: {
    assert(MI.getOperand(3).isImm());
    int64_t GroupOp = MI.getOperand(3).getImm();
    switch (GroupOp) {
    case SPIRV::GroupOperation::Reduce:
    case SPIRV::GroupOperation::InclusiveScan:
    case SPIRV::GroupOperation::ExclusiveScan:
      Reqs.addCapability(SPIRV::Capability::Kernel);
      Reqs.addCapability(SPIRV::Capability::GroupNonUniformArithmetic);
      Reqs.addCapability(SPIRV::Capability::GroupNonUniformBallot);
      break;
    case SPIRV::GroupOperation::ClusteredReduce:
      Reqs.addCapability(SPIRV::Capability::GroupNonUniformClustered);
      break;
    case SPIRV::GroupOperation::PartitionedReduceNV:
    case SPIRV::GroupOperation::PartitionedInclusiveScanNV:
    case SPIRV::GroupOperation::PartitionedExclusiveScanNV:
      Reqs.addCapability(SPIRV::Capability::GroupNonUniformPartitionedNV);
      break;
    }
    break;
  }
  case SPIRV::OpGroupNonUniformShuffle:
  case SPIRV::OpGroupNonUniformShuffleXor:
    Reqs.addCapability(SPIRV::Capability::GroupNonUniformShuffle);
    break;
  case SPIRV::OpGroupNonUniformShuffleUp:
  case SPIRV::OpGroupNonUniformShuffleDown:
    Reqs.addCapability(SPIRV::Capability::GroupNonUniformShuffleRelative);
    break;
  case SPIRV::OpGroupAll:
  case SPIRV::OpGroupAny:
  case SPIRV::OpGroupBroadcast:
  case SPIRV::OpGroupIAdd:
  case SPIRV::OpGroupFAdd:
  case SPIRV::OpGroupFMin:
  case SPIRV::OpGroupUMin:
  case SPIRV::OpGroupSMin:
  case SPIRV::OpGroupFMax:
  case SPIRV::OpGroupUMax:
  case SPIRV::OpGroupSMax:
    Reqs.addCapability(SPIRV::Capability::Groups);
    break;
  case SPIRV::OpGroupNonUniformElect:
    Reqs.addCapability(SPIRV::Capability::GroupNonUniform);
    break;
  case SPIRV::OpGroupNonUniformAll:
  case SPIRV::OpGroupNonUniformAny:
  case SPIRV::OpGroupNonUniformAllEqual:
    Reqs.addCapability(SPIRV::Capability::GroupNonUniformVote);
    break;
  case SPIRV::OpGroupNonUniformBroadcast:
  case SPIRV::OpGroupNonUniformBroadcastFirst:
  case SPIRV::OpGroupNonUniformBallot:
  case SPIRV::OpGroupNonUniformInverseBallot:
  case SPIRV::OpGroupNonUniformBallotBitExtract:
  case SPIRV::OpGroupNonUniformBallotBitCount:
  case SPIRV::OpGroupNonUniformBallotFindLSB:
  case SPIRV::OpGroupNonUniformBallotFindMSB:
    Reqs.addCapability(SPIRV::Capability::GroupNonUniformBallot);
    break;
  default:
    break;
  }
}

static void collectReqs(const Module &M, SPIRV::ModuleAnalysisInfo &MAI,
                        MachineModuleInfo *MMI, const SPIRVSubtarget &ST) {
  // Collect requirements for existing instructions.
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    for (const MachineBasicBlock &MBB : *MF)
      for (const MachineInstr &MI : MBB)
        addInstrRequirements(MI, MAI.Reqs, ST);
  }
  // Collect requirements for OpExecutionMode instructions.
  auto Node = M.getNamedMetadata("spirv.ExecutionMode");
  if (Node) {
    for (unsigned i = 0; i < Node->getNumOperands(); i++) {
      MDNode *MDN = cast<MDNode>(Node->getOperand(i));
      const MDOperand &MDOp = MDN->getOperand(1);
      if (auto *CMeta = dyn_cast<ConstantAsMetadata>(MDOp)) {
        Constant *C = CMeta->getValue();
        if (ConstantInt *Const = dyn_cast<ConstantInt>(C)) {
          auto EM = Const->getZExtValue();
          MAI.Reqs.getAndAddRequirements(
              SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
        }
      }
    }
  }
  for (auto FI = M.begin(), E = M.end(); FI != E; ++FI) {
    const Function &F = *FI;
    if (F.isDeclaration())
      continue;
    if (F.getMetadata("reqd_work_group_size"))
      MAI.Reqs.getAndAddRequirements(
          SPIRV::OperandCategory::ExecutionModeOperand,
          SPIRV::ExecutionMode::LocalSize, ST);
    if (F.getFnAttribute("hlsl.numthreads").isValid()) {
      MAI.Reqs.getAndAddRequirements(
          SPIRV::OperandCategory::ExecutionModeOperand,
          SPIRV::ExecutionMode::LocalSize, ST);
    }
    if (F.getMetadata("work_group_size_hint"))
      MAI.Reqs.getAndAddRequirements(
          SPIRV::OperandCategory::ExecutionModeOperand,
          SPIRV::ExecutionMode::LocalSizeHint, ST);
    if (F.getMetadata("intel_reqd_sub_group_size"))
      MAI.Reqs.getAndAddRequirements(
          SPIRV::OperandCategory::ExecutionModeOperand,
          SPIRV::ExecutionMode::SubgroupSize, ST);
    if (F.getMetadata("vec_type_hint"))
      MAI.Reqs.getAndAddRequirements(
          SPIRV::OperandCategory::ExecutionModeOperand,
          SPIRV::ExecutionMode::VecTypeHint, ST);

    if (F.hasOptNone() &&
        ST.canUseExtension(SPIRV::Extension::SPV_INTEL_optnone)) {
      // Output OpCapability OptNoneINTEL.
      MAI.Reqs.addExtension(SPIRV::Extension::SPV_INTEL_optnone);
      MAI.Reqs.addCapability(SPIRV::Capability::OptNoneINTEL);
    }
  }
}

static unsigned getFastMathFlags(const MachineInstr &I) {
  unsigned Flags = SPIRV::FPFastMathMode::None;
  if (I.getFlag(MachineInstr::MIFlag::FmNoNans))
    Flags |= SPIRV::FPFastMathMode::NotNaN;
  if (I.getFlag(MachineInstr::MIFlag::FmNoInfs))
    Flags |= SPIRV::FPFastMathMode::NotInf;
  if (I.getFlag(MachineInstr::MIFlag::FmNsz))
    Flags |= SPIRV::FPFastMathMode::NSZ;
  if (I.getFlag(MachineInstr::MIFlag::FmArcp))
    Flags |= SPIRV::FPFastMathMode::AllowRecip;
  if (I.getFlag(MachineInstr::MIFlag::FmReassoc))
    Flags |= SPIRV::FPFastMathMode::Fast;
  return Flags;
}

static void handleMIFlagDecoration(MachineInstr &I, const SPIRVSubtarget &ST,
                                   const SPIRVInstrInfo &TII,
                                   SPIRV::RequirementHandler &Reqs) {
  if (I.getFlag(MachineInstr::MIFlag::NoSWrap) && TII.canUseNSW(I) &&
      getSymbolicOperandRequirements(SPIRV::OperandCategory::DecorationOperand,
                                     SPIRV::Decoration::NoSignedWrap, ST, Reqs)
          .IsSatisfiable) {
    buildOpDecorate(I.getOperand(0).getReg(), I, TII,
                    SPIRV::Decoration::NoSignedWrap, {});
  }
  if (I.getFlag(MachineInstr::MIFlag::NoUWrap) && TII.canUseNUW(I) &&
      getSymbolicOperandRequirements(SPIRV::OperandCategory::DecorationOperand,
                                     SPIRV::Decoration::NoUnsignedWrap, ST,
                                     Reqs)
          .IsSatisfiable) {
    buildOpDecorate(I.getOperand(0).getReg(), I, TII,
                    SPIRV::Decoration::NoUnsignedWrap, {});
  }
  if (!TII.canUseFastMathFlags(I))
    return;
  unsigned FMFlags = getFastMathFlags(I);
  if (FMFlags == SPIRV::FPFastMathMode::None)
    return;
  Register DstReg = I.getOperand(0).getReg();
  buildOpDecorate(DstReg, I, TII, SPIRV::Decoration::FPFastMathMode, {FMFlags});
}

// Walk all functions and add decorations related to MI flags.
static void addDecorations(const Module &M, const SPIRVInstrInfo &TII,
                           MachineModuleInfo *MMI, const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    for (auto &MBB : *MF)
      for (auto &MI : MBB)
        handleMIFlagDecoration(MI, ST, TII, MAI.Reqs);
  }
}

struct SPIRV::ModuleAnalysisInfo SPIRVModuleAnalysis::MAI;

void SPIRVModuleAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<MachineModuleInfoWrapperPass>();
}

bool SPIRVModuleAnalysis::runOnModule(Module &M) {
  SPIRVTargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<SPIRVTargetMachine>();
  ST = TM.getSubtargetImpl();
  GR = ST->getSPIRVGlobalRegistry();
  TII = ST->getInstrInfo();

  MMI = &getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  setBaseInfo(M);

  addDecorations(M, *TII, MMI, *ST, MAI);

  collectReqs(M, MAI, MMI, *ST);

  // Process type/const/global var/func decl instructions, number their
  // destination registers from 0 to N, collect Extensions and Capabilities.
  processDefInstrs(M);

  // Number rest of registers from N+1 onwards.
  numberRegistersGlobally(M);

  // Collect OpName, OpEntryPoint, OpDecorate etc, process other instructions.
  processOtherInstrs(M);

  // If there are no entry points, we need the Linkage capability.
  if (MAI.MS[SPIRV::MB_EntryPoints].empty())
    MAI.Reqs.addCapability(SPIRV::Capability::Linkage);

  return false;
}
