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
#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-module-analysis"

static cl::opt<bool>
    SPVDumpDeps("spv-dump-deps",
                cl::desc("Dump MIR with SPIR-V dependencies info"),
                cl::Optional, cl::init(false));

static cl::list<SPIRV::Capability::Capability>
    AvoidCapabilities("avoid-spirv-capabilities",
                      cl::desc("SPIR-V capabilities to avoid if there are "
                               "other options enabling a feature"),
                      cl::ZeroOrMore, cl::Hidden,
                      cl::values(clEnumValN(SPIRV::Capability::Shader, "Shader",
                                            "SPIR-V Shader capability")));
// Use sets instead of cl::list to check "if contains" condition
struct AvoidCapabilitiesSet {
  SmallSet<SPIRV::Capability::Capability, 4> S;
  AvoidCapabilitiesSet() { S.insert_range(AvoidCapabilities); }
};

char llvm::SPIRVModuleAnalysis::ID = 0;

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
  // A set of capabilities to avoid if there is another option.
  AvoidCapabilitiesSet AvoidCaps;
  if (!ST.isShader())
    AvoidCaps.S.insert(SPIRV::Capability::Shader);
  else
    AvoidCaps.S.insert(SPIRV::Capability::Kernel);

  VersionTuple ReqMinVer = getSymbolicOperandMinVersion(Category, i);
  VersionTuple ReqMaxVer = getSymbolicOperandMaxVersion(Category, i);
  VersionTuple SPIRVVersion = ST.getSPIRVVersion();
  bool MinVerOK = SPIRVVersion.empty() || SPIRVVersion >= ReqMinVer;
  bool MaxVerOK =
      ReqMaxVer.empty() || SPIRVVersion.empty() || SPIRVVersion <= ReqMaxVer;
  CapabilityList ReqCaps = getSymbolicOperandCapabilities(Category, i);
  ExtensionList ReqExts = getSymbolicOperandExtensions(Category, i);
  if (ReqCaps.empty()) {
    if (ReqExts.empty()) {
      if (MinVerOK && MaxVerOK)
        return {true, {}, {}, ReqMinVer, ReqMaxVer};
      return {false, {}, {}, VersionTuple(), VersionTuple()};
    }
  } else if (MinVerOK && MaxVerOK) {
    if (ReqCaps.size() == 1) {
      auto Cap = ReqCaps[0];
      if (Reqs.isCapabilityAvailable(Cap)) {
        ReqExts.append(getSymbolicOperandExtensions(
            SPIRV::OperandCategory::CapabilityOperand, Cap));
        return {true, {Cap}, std::move(ReqExts), ReqMinVer, ReqMaxVer};
      }
    } else {
      // By SPIR-V specification: "If an instruction, enumerant, or other
      // feature specifies multiple enabling capabilities, only one such
      // capability needs to be declared to use the feature." However, one
      // capability may be preferred over another. We use command line
      // argument(s) and AvoidCapabilities to avoid selection of certain
      // capabilities if there are other options.
      CapabilityList UseCaps;
      for (auto Cap : ReqCaps)
        if (Reqs.isCapabilityAvailable(Cap))
          UseCaps.push_back(Cap);
      for (size_t i = 0, Sz = UseCaps.size(); i < Sz; ++i) {
        auto Cap = UseCaps[i];
        if (i == Sz - 1 || !AvoidCaps.S.contains(Cap)) {
          ReqExts.append(getSymbolicOperandExtensions(
              SPIRV::OperandCategory::CapabilityOperand, Cap));
          return {true, {Cap}, std::move(ReqExts), ReqMinVer, ReqMaxVer};
        }
      }
    }
  }
  // If there are no capabilities, or we can't satisfy the version or
  // capability requirements, use the list of extensions (if the subtarget
  // can handle them all).
  if (llvm::all_of(ReqExts, [&ST](const SPIRV::Extension::Extension &Ext) {
        return ST.canUseExtension(Ext);
      })) {
    return {true,
            {},
            std::move(ReqExts),
            VersionTuple(),
            VersionTuple()}; // TODO: add versions to extensions.
  }
  return {false, {}, {}, VersionTuple(), VersionTuple()};
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
    MAI.Mem = ST->isShader() ? SPIRV::MemoryModel::GLSL450
                             : SPIRV::MemoryModel::OpenCL;
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
    // Prevent Major part of OpenCL version to be 0
    MAI.SrcLangVersion =
        (std::max(1U, MajorNum) * 100 + MinorNum) * 1000 + RevNum;
  } else {
    // If there is no information about OpenCL version we are forced to generate
    // OpenCL 1.0 by default for the OpenCL environment to avoid puzzling
    // run-times with Unknown/0.0 version output. For a reference, LLVM-SPIRV
    // Translator avoids potential issues with run-times in a similar manner.
    if (!ST->isShader()) {
      MAI.SrcLang = SPIRV::SourceLanguage::OpenCL_CPP;
      MAI.SrcLangVersion = 100000;
    } else {
      MAI.SrcLang = SPIRV::SourceLanguage::Unknown;
      MAI.SrcLangVersion = 0;
    }
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

  if (!ST->isShader()) {
    // TODO: check if it's required by default.
    MAI.ExtInstSetMap[static_cast<unsigned>(
        SPIRV::InstructionSet::OpenCL_std)] = MAI.getNextIDRegister();
  }
}

// Appends the signature of the decoration instructions that decorate R to
// Signature.
static void appendDecorationsForReg(const MachineRegisterInfo &MRI, Register R,
                                    InstrSignature &Signature) {
  for (MachineInstr &UseMI : MRI.use_instructions(R)) {
    // We don't handle OpDecorateId because getting the register alias for the
    // ID can cause problems, and we do not need it for now.
    if (UseMI.getOpcode() != SPIRV::OpDecorate &&
        UseMI.getOpcode() != SPIRV::OpMemberDecorate)
      continue;

    for (unsigned I = 0; I < UseMI.getNumOperands(); ++I) {
      const MachineOperand &MO = UseMI.getOperand(I);
      if (MO.isReg())
        continue;
      Signature.push_back(hash_value(MO));
    }
  }
}

// Returns a representation of an instruction as a vector of MachineOperand
// hash values, see llvm::hash_value(const MachineOperand &MO) for details.
// This creates a signature of the instruction with the same content
// that MachineOperand::isIdenticalTo uses for comparison.
static InstrSignature instrToSignature(const MachineInstr &MI,
                                       SPIRV::ModuleAnalysisInfo &MAI,
                                       bool UseDefReg) {
  Register DefReg;
  InstrSignature Signature{MI.getOpcode()};
  for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
    // The only decorations that can be applied more than once to a given <id>
    // or structure member are UserSemantic(5635), CacheControlLoadINTEL (6442),
    // and CacheControlStoreINTEL (6443). For all the rest of decorations, we
    // will only add to the signature the Opcode, the id to which it applies,
    // and the decoration id, disregarding any decoration flags. This will
    // ensure that any subsequent decoration with the same id will be deemed as
    // a duplicate. Then, at the call site, we will be able to handle duplicates
    // in the best way.
    unsigned Opcode = MI.getOpcode();
    if ((Opcode == SPIRV::OpDecorate) && i >= 2) {
      unsigned DecorationID = MI.getOperand(1).getImm();
      if (DecorationID != SPIRV::Decoration::UserSemantic &&
          DecorationID != SPIRV::Decoration::CacheControlLoadINTEL &&
          DecorationID != SPIRV::Decoration::CacheControlStoreINTEL)
        continue;
    }
    const MachineOperand &MO = MI.getOperand(i);
    size_t h;
    if (MO.isReg()) {
      if (!UseDefReg && MO.isDef()) {
        assert(!DefReg.isValid() && "Multiple def registers.");
        DefReg = MO.getReg();
        continue;
      }
      Register RegAlias = MAI.getRegisterAlias(MI.getMF(), MO.getReg());
      if (!RegAlias.isValid()) {
        LLVM_DEBUG({
          dbgs() << "Unexpectedly, no global id found for the operand ";
          MO.print(dbgs());
          dbgs() << "\nInstruction: ";
          MI.print(dbgs());
          dbgs() << "\n";
        });
        report_fatal_error("All v-regs must have been mapped to global id's");
      }
      // mimic llvm::hash_value(const MachineOperand &MO)
      h = hash_combine(MO.getType(), (unsigned)RegAlias, MO.getSubReg(),
                       MO.isDef());
    } else {
      h = hash_value(MO);
    }
    Signature.push_back(h);
  }

  if (DefReg.isValid()) {
    // Decorations change the semantics of the current instruction. So two
    // identical instruction with different decorations cannot be merged. That
    // is why we add the decorations to the signature.
    appendDecorationsForReg(MI.getMF()->getRegInfo(), DefReg, Signature);
  }
  return Signature;
}

bool SPIRVModuleAnalysis::isDeclSection(const MachineRegisterInfo &MRI,
                                        const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case SPIRV::OpTypeForwardPointer:
    // omit now, collect later
    return false;
  case SPIRV::OpVariable:
    return static_cast<SPIRV::StorageClass::StorageClass>(
               MI.getOperand(2).getImm()) != SPIRV::StorageClass::Function;
  case SPIRV::OpFunction:
  case SPIRV::OpFunctionParameter:
    return true;
  }
  if (GR->hasConstFunPtr() && Opcode == SPIRV::OpUndef) {
    Register DefReg = MI.getOperand(0).getReg();
    for (MachineInstr &UseMI : MRI.use_instructions(DefReg)) {
      if (UseMI.getOpcode() != SPIRV::OpConstantFunctionPointerINTEL)
        continue;
      // it's a dummy definition, FP constant refers to a function,
      // and this is resolved in another way; let's skip this definition
      assert(UseMI.getOperand(2).isReg() &&
             UseMI.getOperand(2).getReg() == DefReg);
      MAI.setSkipEmission(&MI);
      return false;
    }
  }
  return TII->isTypeDeclInstr(MI) || TII->isConstantInstr(MI) ||
         TII->isInlineAsmDefInstr(MI);
}

// This is a special case of a function pointer refering to a possibly
// forward function declaration. The operand is a dummy OpUndef that
// requires a special treatment.
void SPIRVModuleAnalysis::visitFunPtrUse(
    Register OpReg, InstrGRegsMap &SignatureToGReg,
    std::map<const Value *, unsigned> &GlobalToGReg, const MachineFunction *MF,
    const MachineInstr &MI) {
  const MachineOperand *OpFunDef =
      GR->getFunctionDefinitionByUse(&MI.getOperand(2));
  assert(OpFunDef && OpFunDef->isReg());
  // find the actual function definition and number it globally in advance
  const MachineInstr *OpDefMI = OpFunDef->getParent();
  assert(OpDefMI && OpDefMI->getOpcode() == SPIRV::OpFunction);
  const MachineFunction *FunDefMF = OpDefMI->getParent()->getParent();
  const MachineRegisterInfo &FunDefMRI = FunDefMF->getRegInfo();
  do {
    visitDecl(FunDefMRI, SignatureToGReg, GlobalToGReg, FunDefMF, *OpDefMI);
    OpDefMI = OpDefMI->getNextNode();
  } while (OpDefMI && (OpDefMI->getOpcode() == SPIRV::OpFunction ||
                       OpDefMI->getOpcode() == SPIRV::OpFunctionParameter));
  // associate the function pointer with the newly assigned global number
  MCRegister GlobalFunDefReg =
      MAI.getRegisterAlias(FunDefMF, OpFunDef->getReg());
  assert(GlobalFunDefReg.isValid() &&
         "Function definition must refer to a global register");
  MAI.setRegisterAlias(MF, OpReg, GlobalFunDefReg);
}

// Depth first recursive traversal of dependencies. Repeated visits are guarded
// by MAI.hasRegisterAlias().
void SPIRVModuleAnalysis::visitDecl(
    const MachineRegisterInfo &MRI, InstrGRegsMap &SignatureToGReg,
    std::map<const Value *, unsigned> &GlobalToGReg, const MachineFunction *MF,
    const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();

  // Process each operand of the instruction to resolve dependencies
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || MO.isDef())
      continue;
    Register OpReg = MO.getReg();
    // Handle function pointers special case
    if (Opcode == SPIRV::OpConstantFunctionPointerINTEL &&
        MRI.getRegClass(OpReg) == &SPIRV::pIDRegClass) {
      visitFunPtrUse(OpReg, SignatureToGReg, GlobalToGReg, MF, MI);
      continue;
    }
    // Skip already processed instructions
    if (MAI.hasRegisterAlias(MF, MO.getReg()))
      continue;
    // Recursively visit dependencies
    if (const MachineInstr *OpDefMI = MRI.getUniqueVRegDef(OpReg)) {
      if (isDeclSection(MRI, *OpDefMI))
        visitDecl(MRI, SignatureToGReg, GlobalToGReg, MF, *OpDefMI);
      continue;
    }
    // Handle the unexpected case of no unique definition for the SPIR-V
    // instruction
    LLVM_DEBUG({
      dbgs() << "Unexpectedly, no unique definition for the operand ";
      MO.print(dbgs());
      dbgs() << "\nInstruction: ";
      MI.print(dbgs());
      dbgs() << "\n";
    });
    report_fatal_error(
        "No unique definition is found for the virtual register");
  }

  MCRegister GReg;
  bool IsFunDef = false;
  if (TII->isSpecConstantInstr(MI)) {
    GReg = MAI.getNextIDRegister();
    MAI.MS[SPIRV::MB_TypeConstVars].push_back(&MI);
  } else if (Opcode == SPIRV::OpFunction ||
             Opcode == SPIRV::OpFunctionParameter) {
    GReg = handleFunctionOrParameter(MF, MI, GlobalToGReg, IsFunDef);
  } else if (Opcode == SPIRV::OpTypeStruct ||
             Opcode == SPIRV::OpConstantComposite) {
    GReg = handleTypeDeclOrConstant(MI, SignatureToGReg);
    const MachineInstr *NextInstr = MI.getNextNode();
    while (NextInstr &&
           ((Opcode == SPIRV::OpTypeStruct &&
             NextInstr->getOpcode() == SPIRV::OpTypeStructContinuedINTEL) ||
            (Opcode == SPIRV::OpConstantComposite &&
             NextInstr->getOpcode() ==
                 SPIRV::OpConstantCompositeContinuedINTEL))) {
      MCRegister Tmp = handleTypeDeclOrConstant(*NextInstr, SignatureToGReg);
      MAI.setRegisterAlias(MF, NextInstr->getOperand(0).getReg(), Tmp);
      MAI.setSkipEmission(NextInstr);
      NextInstr = NextInstr->getNextNode();
    }
  } else if (TII->isTypeDeclInstr(MI) || TII->isConstantInstr(MI) ||
             TII->isInlineAsmDefInstr(MI)) {
    GReg = handleTypeDeclOrConstant(MI, SignatureToGReg);
  } else if (Opcode == SPIRV::OpVariable) {
    GReg = handleVariable(MF, MI, GlobalToGReg);
  } else {
    LLVM_DEBUG({
      dbgs() << "\nInstruction: ";
      MI.print(dbgs());
      dbgs() << "\n";
    });
    llvm_unreachable("Unexpected instruction is visited");
  }
  MAI.setRegisterAlias(MF, MI.getOperand(0).getReg(), GReg);
  if (!IsFunDef)
    MAI.setSkipEmission(&MI);
}

MCRegister SPIRVModuleAnalysis::handleFunctionOrParameter(
    const MachineFunction *MF, const MachineInstr &MI,
    std::map<const Value *, unsigned> &GlobalToGReg, bool &IsFunDef) {
  const Value *GObj = GR->getGlobalObject(MF, MI.getOperand(0).getReg());
  assert(GObj && "Unregistered global definition");
  const Function *F = dyn_cast<Function>(GObj);
  if (!F)
    F = dyn_cast<Argument>(GObj)->getParent();
  assert(F && "Expected a reference to a function or an argument");
  IsFunDef = !F->isDeclaration();
  auto [It, Inserted] = GlobalToGReg.try_emplace(GObj);
  if (!Inserted)
    return It->second;
  MCRegister GReg = MAI.getNextIDRegister();
  It->second = GReg;
  if (!IsFunDef)
    MAI.MS[SPIRV::MB_ExtFuncDecls].push_back(&MI);
  return GReg;
}

MCRegister
SPIRVModuleAnalysis::handleTypeDeclOrConstant(const MachineInstr &MI,
                                              InstrGRegsMap &SignatureToGReg) {
  InstrSignature MISign = instrToSignature(MI, MAI, false);
  auto [It, Inserted] = SignatureToGReg.try_emplace(MISign);
  if (!Inserted)
    return It->second;
  MCRegister GReg = MAI.getNextIDRegister();
  It->second = GReg;
  MAI.MS[SPIRV::MB_TypeConstVars].push_back(&MI);
  return GReg;
}

MCRegister SPIRVModuleAnalysis::handleVariable(
    const MachineFunction *MF, const MachineInstr &MI,
    std::map<const Value *, unsigned> &GlobalToGReg) {
  MAI.GlobalVarList.push_back(&MI);
  const Value *GObj = GR->getGlobalObject(MF, MI.getOperand(0).getReg());
  assert(GObj && "Unregistered global definition");
  auto [It, Inserted] = GlobalToGReg.try_emplace(GObj);
  if (!Inserted)
    return It->second;
  MCRegister GReg = MAI.getNextIDRegister();
  It->second = GReg;
  MAI.MS[SPIRV::MB_TypeConstVars].push_back(&MI);
  return GReg;
}

void SPIRVModuleAnalysis::collectDeclarations(const Module &M) {
  InstrGRegsMap SignatureToGReg;
  std::map<const Value *, unsigned> GlobalToGReg;
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    const MachineRegisterInfo &MRI = MF->getRegInfo();
    unsigned PastHeader = 0;
    for (MachineBasicBlock &MBB : *MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.getNumOperands() == 0)
          continue;
        unsigned Opcode = MI.getOpcode();
        if (Opcode == SPIRV::OpFunction) {
          if (PastHeader == 0) {
            PastHeader = 1;
            continue;
          }
        } else if (Opcode == SPIRV::OpFunctionParameter) {
          if (PastHeader < 2)
            continue;
        } else if (PastHeader > 0) {
          PastHeader = 2;
        }

        const MachineOperand &DefMO = MI.getOperand(0);
        switch (Opcode) {
        case SPIRV::OpExtension:
          MAI.Reqs.addExtension(SPIRV::Extension::Extension(DefMO.getImm()));
          MAI.setSkipEmission(&MI);
          break;
        case SPIRV::OpCapability:
          MAI.Reqs.addCapability(SPIRV::Capability::Capability(DefMO.getImm()));
          MAI.setSkipEmission(&MI);
          if (PastHeader > 0)
            PastHeader = 2;
          break;
        default:
          if (DefMO.isReg() && isDeclSection(MRI, MI) &&
              !MAI.hasRegisterAlias(MF, DefMO.getReg()))
            visitDecl(MRI, SignatureToGReg, GlobalToGReg, MF, MI);
        }
      }
    }
  }
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
    if (Dec == SPIRV::Decoration::LinkageAttributes) {
      auto Lnk = MI.getOperand(MI.getNumOperands() - 1).getImm();
      if (Lnk == SPIRV::LinkageType::Import) {
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
    MCRegister GlobalReg = MAI.getRegisterAlias(MI.getMF(), Reg);
    assert(GlobalReg.isValid());
    MAI.FuncMap[F] = GlobalReg;
  }
}

// Collect the given instruction in the specified MS. We assume global register
// numbering has already occurred by this point. We can directly compare reg
// arguments when detecting duplicates.
static void collectOtherInstr(MachineInstr &MI, SPIRV::ModuleAnalysisInfo &MAI,
                              SPIRV::ModuleSectionType MSType, InstrTraces &IS,
                              bool Append = true) {
  MAI.setSkipEmission(&MI);
  InstrSignature MISign = instrToSignature(MI, MAI, true);
  auto FoundMI = IS.insert(std::move(MISign));
  if (!FoundMI.second) {
    if (MI.getOpcode() == SPIRV::OpDecorate) {
      assert(MI.getNumOperands() >= 2 &&
             "Decoration instructions must have at least 2 operands");
      assert(MSType == SPIRV::MB_Annotations &&
             "Only OpDecorate instructions can be duplicates");
      // For FPFastMathMode decoration, we need to merge the flags of the
      // duplicate decoration with the original one, so we need to find the
      // original instruction that has the same signature. For the rest of
      // instructions, we will simply skip the duplicate.
      if (MI.getOperand(1).getImm() != SPIRV::Decoration::FPFastMathMode)
        return; // Skip duplicates of other decorations.

      const SPIRV::InstrList &Decorations = MAI.MS[MSType];
      for (const MachineInstr *OrigMI : Decorations) {
        if (instrToSignature(*OrigMI, MAI, true) == MISign) {
          assert(OrigMI->getNumOperands() == MI.getNumOperands() &&
                 "Original instruction must have the same number of operands");
          assert(
              OrigMI->getNumOperands() == 3 &&
              "FPFastMathMode decoration must have 3 operands for OpDecorate");
          unsigned OrigFlags = OrigMI->getOperand(2).getImm();
          unsigned NewFlags = MI.getOperand(2).getImm();
          if (OrigFlags == NewFlags)
            return; // No need to merge, the flags are the same.

          // Emit warning about possible conflict between flags.
          unsigned FinalFlags = OrigFlags | NewFlags;
          llvm::errs()
              << "Warning: Conflicting FPFastMathMode decoration flags "
                 "in instruction: "
              << *OrigMI << "Original flags: " << OrigFlags
              << ", new flags: " << NewFlags
              << ". They will be merged on a best effort basis, but not "
                 "validated. Final flags: "
              << FinalFlags << "\n";
          MachineInstr *OrigMINonConst = const_cast<MachineInstr *>(OrigMI);
          MachineOperand &OrigFlagsOp = OrigMINonConst->getOperand(2);
          OrigFlagsOp =
              MachineOperand::CreateImm(static_cast<unsigned>(FinalFlags));
          return; // Merge done, so we found a duplicate; don't add it to MAI.MS
        }
      }
      assert(false && "No original instruction found for the duplicate "
                      "OpDecorate, but we found one in IS.");
    }
    return; // insert failed, so we found a duplicate; don't add it to MAI.MS
  }
  // No duplicates, so add it.
  if (Append)
    MAI.MS[MSType].push_back(&MI);
  else
    MAI.MS[MSType].insert(MAI.MS[MSType].begin(), &MI);
}

// Some global instructions make reference to function-local ID regs, so cannot
// be correctly collected until these registers are globally numbered.
void SPIRVModuleAnalysis::processOtherInstrs(const Module &M) {
  InstrTraces IS;
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
        if (OpCode == SPIRV::OpString) {
          collectOtherInstr(MI, MAI, SPIRV::MB_DebugStrings, IS);
        } else if (OpCode == SPIRV::OpExtInst && MI.getOperand(2).isImm() &&
                   MI.getOperand(2).getImm() ==
                       SPIRV::InstructionSet::
                           NonSemantic_Shader_DebugInfo_100) {
          MachineOperand Ins = MI.getOperand(3);
          namespace NS = SPIRV::NonSemanticExtInst;
          static constexpr int64_t GlobalNonSemanticDITy[] = {
              NS::DebugSource, NS::DebugCompilationUnit, NS::DebugInfoNone,
              NS::DebugTypeBasic, NS::DebugTypePointer};
          bool IsGlobalDI = false;
          for (unsigned Idx = 0; Idx < std::size(GlobalNonSemanticDITy); ++Idx)
            IsGlobalDI |= Ins.getImm() == GlobalNonSemanticDITy[Idx];
          if (IsGlobalDI)
            collectOtherInstr(MI, MAI, SPIRV::MB_NonSemanticGlobalDI, IS);
        } else if (OpCode == SPIRV::OpName || OpCode == SPIRV::OpMemberName) {
          collectOtherInstr(MI, MAI, SPIRV::MB_DebugNames, IS);
        } else if (OpCode == SPIRV::OpEntryPoint) {
          collectOtherInstr(MI, MAI, SPIRV::MB_EntryPoints, IS);
        } else if (TII->isAliasingInstr(MI)) {
          collectOtherInstr(MI, MAI, SPIRV::MB_AliasingInsts, IS);
        } else if (TII->isDecorationInstr(MI)) {
          collectOtherInstr(MI, MAI, SPIRV::MB_Annotations, IS);
          collectFuncNames(MI, &*F);
        } else if (TII->isConstantInstr(MI)) {
          // Now OpSpecConstant*s are not in DT,
          // but they need to be collected anyway.
          collectOtherInstr(MI, MAI, SPIRV::MB_TypeConstVars, IS);
        } else if (OpCode == SPIRV::OpFunction) {
          collectFuncNames(MI, &*F);
        } else if (OpCode == SPIRV::OpTypeForwardPointer) {
          collectOtherInstr(MI, MAI, SPIRV::MB_TypeConstVars, IS, false);
        }
      }
  }
}

// Number registers in all functions globally from 0 onwards and store
// the result in global register alias table. Some registers are already
// numbered.
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
          MCRegister NewReg = MAI.getNextIDRegister();
          MAI.setRegisterAlias(MF, Reg, NewReg);
        }
        if (MI.getOpcode() != SPIRV::OpExtInst)
          continue;
        auto Set = MI.getOperand(2).getImm();
        auto [It, Inserted] = MAI.ExtInstSetMap.try_emplace(Set);
        if (Inserted)
          It->second = MAI.getNextIDRegister();
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

void SPIRV::RequirementHandler::recursiveAddCapabilities(
    const CapabilityList &ToPrune) {
  for (const auto &Cap : ToPrune) {
    AllCaps.insert(Cap);
    CapabilityList ImplicitDecls =
        getSymbolicOperandCapabilities(OperandCategory::CapabilityOperand, Cap);
    recursiveAddCapabilities(ImplicitDecls);
  }
}

void SPIRV::RequirementHandler::addCapabilities(const CapabilityList &ToAdd) {
  for (const auto &Cap : ToAdd) {
    bool IsNewlyInserted = AllCaps.insert(Cap).second;
    if (!IsNewlyInserted) // Don't re-add if it's already been declared.
      continue;
    CapabilityList ImplicitDecls =
        getSymbolicOperandCapabilities(OperandCategory::CapabilityOperand, Cap);
    recursiveAddCapabilities(ImplicitDecls);
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

  if (!Req.MinVer.empty()) {
    if (!MaxVersion.empty() && Req.MinVer > MaxVersion) {
      LLVM_DEBUG(dbgs() << "Conflicting version requirements: >= " << Req.MinVer
                        << " and <= " << MaxVersion << "\n");
      report_fatal_error("Adding SPIR-V requirements that can't be satisfied.");
    }

    if (MinVersion.empty() || Req.MinVer > MinVersion)
      MinVersion = Req.MinVer;
  }

  if (!Req.MaxVer.empty()) {
    if (!MinVersion.empty() && Req.MaxVer < MinVersion) {
      LLVM_DEBUG(dbgs() << "Conflicting version requirements: <= " << Req.MaxVer
                        << " and >= " << MinVersion << "\n");
      report_fatal_error("Adding SPIR-V requirements that can't be satisfied.");
    }

    if (MaxVersion.empty() || Req.MaxVer < MaxVersion)
      MaxVersion = Req.MaxVer;
  }
}

void SPIRV::RequirementHandler::checkSatisfiable(
    const SPIRVSubtarget &ST) const {
  // Report as many errors as possible before aborting the compilation.
  bool IsSatisfiable = true;
  auto TargetVer = ST.getSPIRVVersion();

  if (!MaxVersion.empty() && !TargetVer.empty() && MaxVersion < TargetVer) {
    LLVM_DEBUG(
        dbgs() << "Target SPIR-V version too high for required features\n"
               << "Required max version: " << MaxVersion << " target version "
               << TargetVer << "\n");
    IsSatisfiable = false;
  }

  if (!MinVersion.empty() && !TargetVer.empty() && MinVersion > TargetVer) {
    LLVM_DEBUG(dbgs() << "Target SPIR-V version too low for required features\n"
                      << "Required min version: " << MinVersion
                      << " target version " << TargetVer << "\n");
    IsSatisfiable = false;
  }

  if (!MinVersion.empty() && !MaxVersion.empty() && MinVersion > MaxVersion) {
    LLVM_DEBUG(
        dbgs()
        << "Version is too low for some features and too high for others.\n"
        << "Required SPIR-V min version: " << MinVersion
        << " required SPIR-V max version " << MaxVersion << "\n");
    IsSatisfiable = false;
  }

  AvoidCapabilitiesSet AvoidCaps;
  if (!ST.isShader())
    AvoidCaps.S.insert(SPIRV::Capability::Shader);
  else
    AvoidCaps.S.insert(SPIRV::Capability::Kernel);

  for (auto Cap : MinimalCaps) {
    if (AvailableCaps.contains(Cap) && !AvoidCaps.S.contains(Cap))
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
    LLVM_DEBUG(dbgs() << "Extension not supported: "
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

void SPIRV::RequirementHandler::removeCapabilityIf(
    const Capability::Capability ToRemove,
    const Capability::Capability IfPresent) {
  if (AllCaps.contains(IfPresent))
    AllCaps.erase(ToRemove);
}

namespace llvm {
namespace SPIRV {
void RequirementHandler::initAvailableCapabilities(const SPIRVSubtarget &ST) {
  // Provided by both all supported Vulkan versions and OpenCl.
  addAvailableCaps({Capability::Shader, Capability::Linkage, Capability::Int8,
                    Capability::Int16});

  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 3)))
    addAvailableCaps({Capability::GroupNonUniform,
                      Capability::GroupNonUniformVote,
                      Capability::GroupNonUniformArithmetic,
                      Capability::GroupNonUniformBallot,
                      Capability::GroupNonUniformClustered,
                      Capability::GroupNonUniformShuffle,
                      Capability::GroupNonUniformShuffleRelative});

  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 6)))
    addAvailableCaps({Capability::DotProduct, Capability::DotProductInputAll,
                      Capability::DotProductInput4x8Bit,
                      Capability::DotProductInput4x8BitPacked,
                      Capability::DemoteToHelperInvocation});

  // Add capabilities enabled by extensions.
  for (auto Extension : ST.getAllAvailableExtensions()) {
    CapabilityList EnabledCapabilities =
        getCapabilitiesEnabledByExtension(Extension);
    addAvailableCaps(EnabledCapabilities);
  }

  if (!ST.isShader()) {
    initAvailableCapabilitiesForOpenCL(ST);
    return;
  }

  if (ST.isShader()) {
    initAvailableCapabilitiesForVulkan(ST);
    return;
  }

  report_fatal_error("Unimplemented environment for SPIR-V generation.");
}

void RequirementHandler::initAvailableCapabilitiesForOpenCL(
    const SPIRVSubtarget &ST) {
  // Add the min requirements for different OpenCL and SPIR-V versions.
  addAvailableCaps({Capability::Addresses, Capability::Float16Buffer,
                    Capability::Kernel, Capability::Vector16,
                    Capability::Groups, Capability::GenericPointer,
                    Capability::StorageImageWriteWithoutFormat,
                    Capability::StorageImageReadWithoutFormat});
  if (ST.hasOpenCLFullProfile())
    addAvailableCaps({Capability::Int64, Capability::Int64Atomics});
  if (ST.hasOpenCLImageSupport()) {
    addAvailableCaps({Capability::ImageBasic, Capability::LiteralSampler,
                      Capability::Image1D, Capability::SampledBuffer,
                      Capability::ImageBuffer});
    if (ST.isAtLeastOpenCLVer(VersionTuple(2, 0)))
      addAvailableCaps({Capability::ImageReadWrite});
  }
  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 1)) &&
      ST.isAtLeastOpenCLVer(VersionTuple(2, 2)))
    addAvailableCaps({Capability::SubgroupDispatch, Capability::PipeStorage});
  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 4)))
    addAvailableCaps({Capability::DenormPreserve, Capability::DenormFlushToZero,
                      Capability::SignedZeroInfNanPreserve,
                      Capability::RoundingModeRTE,
                      Capability::RoundingModeRTZ});
  // TODO: verify if this needs some checks.
  addAvailableCaps({Capability::Float16, Capability::Float64});

  // TODO: add OpenCL extensions.
}

void RequirementHandler::initAvailableCapabilitiesForVulkan(
    const SPIRVSubtarget &ST) {

  // Core in Vulkan 1.1 and earlier.
  addAvailableCaps({Capability::Int64, Capability::Float16, Capability::Float64,
                    Capability::GroupNonUniform, Capability::Image1D,
                    Capability::SampledBuffer, Capability::ImageBuffer,
                    Capability::UniformBufferArrayDynamicIndexing,
                    Capability::SampledImageArrayDynamicIndexing,
                    Capability::StorageBufferArrayDynamicIndexing,
                    Capability::StorageImageArrayDynamicIndexing});

  // Became core in Vulkan 1.2
  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 5))) {
    addAvailableCaps(
        {Capability::ShaderNonUniformEXT, Capability::RuntimeDescriptorArrayEXT,
         Capability::InputAttachmentArrayDynamicIndexingEXT,
         Capability::UniformTexelBufferArrayDynamicIndexingEXT,
         Capability::StorageTexelBufferArrayDynamicIndexingEXT,
         Capability::UniformBufferArrayNonUniformIndexingEXT,
         Capability::SampledImageArrayNonUniformIndexingEXT,
         Capability::StorageBufferArrayNonUniformIndexingEXT,
         Capability::StorageImageArrayNonUniformIndexingEXT,
         Capability::InputAttachmentArrayNonUniformIndexingEXT,
         Capability::UniformTexelBufferArrayNonUniformIndexingEXT,
         Capability::StorageTexelBufferArrayNonUniformIndexingEXT});
  }

  // Became core in Vulkan 1.3
  if (ST.isAtLeastSPIRVVer(VersionTuple(1, 6)))
    addAvailableCaps({Capability::StorageImageWriteWithoutFormat,
                      Capability::StorageImageReadWithoutFormat});
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
  } else if (Dec == SPIRV::Decoration::LinkageAttributes) {
    int64_t LinkageOp = MI.getOperand(MI.getNumOperands() - 1).getImm();
    SPIRV::LinkageType::LinkageType LnkType =
        static_cast<SPIRV::LinkageType::LinkageType>(LinkageOp);
    if (LnkType == SPIRV::LinkageType::LinkOnceODR)
      Reqs.addExtension(SPIRV::Extension::SPV_KHR_linkonce_odr);
  } else if (Dec == SPIRV::Decoration::CacheControlLoadINTEL ||
             Dec == SPIRV::Decoration::CacheControlStoreINTEL) {
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_cache_controls);
  } else if (Dec == SPIRV::Decoration::HostAccessINTEL) {
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_global_variable_host_access);
  } else if (Dec == SPIRV::Decoration::InitModeINTEL ||
             Dec == SPIRV::Decoration::ImplementInRegisterMapINTEL) {
    Reqs.addExtension(
        SPIRV::Extension::SPV_INTEL_global_variable_fpga_decorations);
  } else if (Dec == SPIRV::Decoration::NonUniformEXT) {
    Reqs.addRequirements(SPIRV::Capability::ShaderNonUniformEXT);
  } else if (Dec == SPIRV::Decoration::FPMaxErrorDecorationINTEL) {
    Reqs.addRequirements(SPIRV::Capability::FPMaxErrorINTEL);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_fp_max_error);
  } else if (Dec == SPIRV::Decoration::FPFastMathMode) {
    if (ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2)) {
      Reqs.addRequirements(SPIRV::Capability::FloatControls2);
      Reqs.addExtension(SPIRV::Extension::SPV_KHR_float_controls2);
    }
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
  if (!ST.isShader()) {
    if (MI.getNumOperands() > 8 &&
        MI.getOperand(8).getImm() == SPIRV::AccessQualifier::ReadWrite)
      Reqs.addRequirements(SPIRV::Capability::ImageReadWrite);
    else
      Reqs.addRequirements(SPIRV::Capability::ImageBasic);
  }
}

// Add requirements for handling atomic float instructions
#define ATOM_FLT_REQ_EXT_MSG(ExtName)                                          \
  "The atomic float instruction requires the following SPIR-V "                \
  "extension: SPV_EXT_shader_atomic_float" ExtName
static void AddAtomicFloatRequirements(const MachineInstr &MI,
                                       SPIRV::RequirementHandler &Reqs,
                                       const SPIRVSubtarget &ST) {
  assert(MI.getOperand(1).isReg() &&
         "Expect register operand in atomic float instruction");
  Register TypeReg = MI.getOperand(1).getReg();
  SPIRVType *TypeDef = MI.getMF()->getRegInfo().getVRegDef(TypeReg);
  if (TypeDef->getOpcode() != SPIRV::OpTypeFloat)
    report_fatal_error("Result type of an atomic float instruction must be a "
                       "floating-point type scalar");

  unsigned BitWidth = TypeDef->getOperand(1).getImm();
  unsigned Op = MI.getOpcode();
  if (Op == SPIRV::OpAtomicFAddEXT) {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_EXT_shader_atomic_float_add))
      report_fatal_error(ATOM_FLT_REQ_EXT_MSG("_add"), false);
    Reqs.addExtension(SPIRV::Extension::SPV_EXT_shader_atomic_float_add);
    switch (BitWidth) {
    case 16:
      if (!ST.canUseExtension(
              SPIRV::Extension::SPV_EXT_shader_atomic_float16_add))
        report_fatal_error(ATOM_FLT_REQ_EXT_MSG("16_add"), false);
      Reqs.addExtension(SPIRV::Extension::SPV_EXT_shader_atomic_float16_add);
      Reqs.addCapability(SPIRV::Capability::AtomicFloat16AddEXT);
      break;
    case 32:
      Reqs.addCapability(SPIRV::Capability::AtomicFloat32AddEXT);
      break;
    case 64:
      Reqs.addCapability(SPIRV::Capability::AtomicFloat64AddEXT);
      break;
    default:
      report_fatal_error(
          "Unexpected floating-point type width in atomic float instruction");
    }
  } else {
    if (!ST.canUseExtension(
            SPIRV::Extension::SPV_EXT_shader_atomic_float_min_max))
      report_fatal_error(ATOM_FLT_REQ_EXT_MSG("_min_max"), false);
    Reqs.addExtension(SPIRV::Extension::SPV_EXT_shader_atomic_float_min_max);
    switch (BitWidth) {
    case 16:
      Reqs.addCapability(SPIRV::Capability::AtomicFloat16MinMaxEXT);
      break;
    case 32:
      Reqs.addCapability(SPIRV::Capability::AtomicFloat32MinMaxEXT);
      break;
    case 64:
      Reqs.addCapability(SPIRV::Capability::AtomicFloat64MinMaxEXT);
      break;
    default:
      report_fatal_error(
          "Unexpected floating-point type width in atomic float instruction");
    }
  }
}

bool isUniformTexelBuffer(MachineInstr *ImageInst) {
  if (ImageInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  uint32_t Dim = ImageInst->getOperand(2).getImm();
  uint32_t Sampled = ImageInst->getOperand(6).getImm();
  return Dim == SPIRV::Dim::DIM_Buffer && Sampled == 1;
}

bool isStorageTexelBuffer(MachineInstr *ImageInst) {
  if (ImageInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  uint32_t Dim = ImageInst->getOperand(2).getImm();
  uint32_t Sampled = ImageInst->getOperand(6).getImm();
  return Dim == SPIRV::Dim::DIM_Buffer && Sampled == 2;
}

bool isSampledImage(MachineInstr *ImageInst) {
  if (ImageInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  uint32_t Dim = ImageInst->getOperand(2).getImm();
  uint32_t Sampled = ImageInst->getOperand(6).getImm();
  return Dim != SPIRV::Dim::DIM_Buffer && Sampled == 1;
}

bool isInputAttachment(MachineInstr *ImageInst) {
  if (ImageInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  uint32_t Dim = ImageInst->getOperand(2).getImm();
  uint32_t Sampled = ImageInst->getOperand(6).getImm();
  return Dim == SPIRV::Dim::DIM_SubpassData && Sampled == 2;
}

bool isStorageImage(MachineInstr *ImageInst) {
  if (ImageInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  uint32_t Dim = ImageInst->getOperand(2).getImm();
  uint32_t Sampled = ImageInst->getOperand(6).getImm();
  return Dim != SPIRV::Dim::DIM_Buffer && Sampled == 2;
}

bool isCombinedImageSampler(MachineInstr *SampledImageInst) {
  if (SampledImageInst->getOpcode() != SPIRV::OpTypeSampledImage)
    return false;

  const MachineRegisterInfo &MRI = SampledImageInst->getMF()->getRegInfo();
  Register ImageReg = SampledImageInst->getOperand(1).getReg();
  auto *ImageInst = MRI.getUniqueVRegDef(ImageReg);
  return isSampledImage(ImageInst);
}

bool hasNonUniformDecoration(Register Reg, const MachineRegisterInfo &MRI) {
  for (const auto &MI : MRI.reg_instructions(Reg)) {
    if (MI.getOpcode() != SPIRV::OpDecorate)
      continue;

    uint32_t Dec = MI.getOperand(1).getImm();
    if (Dec == SPIRV::Decoration::NonUniformEXT)
      return true;
  }
  return false;
}

void addOpAccessChainReqs(const MachineInstr &Instr,
                          SPIRV::RequirementHandler &Handler,
                          const SPIRVSubtarget &Subtarget) {
  const MachineRegisterInfo &MRI = Instr.getMF()->getRegInfo();
  // Get the result type. If it is an image type, then the shader uses
  // descriptor indexing. The appropriate capabilities will be added based
  // on the specifics of the image.
  Register ResTypeReg = Instr.getOperand(1).getReg();
  MachineInstr *ResTypeInst = MRI.getUniqueVRegDef(ResTypeReg);

  assert(ResTypeInst->getOpcode() == SPIRV::OpTypePointer);
  uint32_t StorageClass = ResTypeInst->getOperand(1).getImm();
  if (StorageClass != SPIRV::StorageClass::StorageClass::UniformConstant &&
      StorageClass != SPIRV::StorageClass::StorageClass::Uniform &&
      StorageClass != SPIRV::StorageClass::StorageClass::StorageBuffer) {
    return;
  }

  bool IsNonUniform =
      hasNonUniformDecoration(Instr.getOperand(0).getReg(), MRI);

  auto FirstIndexReg = Instr.getOperand(3).getReg();
  bool FirstIndexIsConstant =
      Subtarget.getInstrInfo()->isConstantInstr(*MRI.getVRegDef(FirstIndexReg));

  if (StorageClass == SPIRV::StorageClass::StorageClass::StorageBuffer) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::StorageBufferArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::StorageBufferArrayDynamicIndexing);
    return;
  }

  Register PointeeTypeReg = ResTypeInst->getOperand(2).getReg();
  MachineInstr *PointeeType = MRI.getUniqueVRegDef(PointeeTypeReg);
  if (PointeeType->getOpcode() != SPIRV::OpTypeImage &&
      PointeeType->getOpcode() != SPIRV::OpTypeSampledImage &&
      PointeeType->getOpcode() != SPIRV::OpTypeSampler) {
    return;
  }

  if (isUniformTexelBuffer(PointeeType)) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::UniformTexelBufferArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::UniformTexelBufferArrayDynamicIndexingEXT);
  } else if (isInputAttachment(PointeeType)) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::InputAttachmentArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::InputAttachmentArrayDynamicIndexingEXT);
  } else if (isStorageTexelBuffer(PointeeType)) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::StorageTexelBufferArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::StorageTexelBufferArrayDynamicIndexingEXT);
  } else if (isSampledImage(PointeeType) ||
             isCombinedImageSampler(PointeeType) ||
             PointeeType->getOpcode() == SPIRV::OpTypeSampler) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::SampledImageArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::SampledImageArrayDynamicIndexing);
  } else if (isStorageImage(PointeeType)) {
    if (IsNonUniform)
      Handler.addRequirements(
          SPIRV::Capability::StorageImageArrayNonUniformIndexingEXT);
    else if (!FirstIndexIsConstant)
      Handler.addRequirements(
          SPIRV::Capability::StorageImageArrayDynamicIndexing);
  }
}

static bool isImageTypeWithUnknownFormat(SPIRVType *TypeInst) {
  if (TypeInst->getOpcode() != SPIRV::OpTypeImage)
    return false;
  assert(TypeInst->getOperand(7).isImm() && "The image format must be an imm.");
  return TypeInst->getOperand(7).getImm() == 0;
}

static void AddDotProductRequirements(const MachineInstr &MI,
                                      SPIRV::RequirementHandler &Reqs,
                                      const SPIRVSubtarget &ST) {
  if (ST.canUseExtension(SPIRV::Extension::SPV_KHR_integer_dot_product))
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_integer_dot_product);
  Reqs.addCapability(SPIRV::Capability::DotProduct);

  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  assert(MI.getOperand(2).isReg() && "Unexpected operand in dot");
  // We do not consider what the previous instruction is. This is just used
  // to get the input register and to check the type.
  const MachineInstr *Input = MRI.getVRegDef(MI.getOperand(2).getReg());
  assert(Input->getOperand(1).isReg() && "Unexpected operand in dot input");
  Register InputReg = Input->getOperand(1).getReg();

  SPIRVType *TypeDef = MRI.getVRegDef(InputReg);
  if (TypeDef->getOpcode() == SPIRV::OpTypeInt) {
    assert(TypeDef->getOperand(1).getImm() == 32);
    Reqs.addCapability(SPIRV::Capability::DotProductInput4x8BitPacked);
  } else if (TypeDef->getOpcode() == SPIRV::OpTypeVector) {
    SPIRVType *ScalarTypeDef = MRI.getVRegDef(TypeDef->getOperand(1).getReg());
    assert(ScalarTypeDef->getOpcode() == SPIRV::OpTypeInt);
    if (ScalarTypeDef->getOperand(1).getImm() == 8) {
      assert(TypeDef->getOperand(2).getImm() == 4 &&
             "Dot operand of 8-bit integer type requires 4 components");
      Reqs.addCapability(SPIRV::Capability::DotProductInput4x8Bit);
    } else {
      Reqs.addCapability(SPIRV::Capability::DotProductInputAll);
    }
  }
}

void addPrintfRequirements(const MachineInstr &MI,
                           SPIRV::RequirementHandler &Reqs,
                           const SPIRVSubtarget &ST) {
  SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  const SPIRVType *PtrType = GR->getSPIRVTypeForVReg(MI.getOperand(4).getReg());
  if (PtrType) {
    MachineOperand ASOp = PtrType->getOperand(1);
    if (ASOp.isImm()) {
      unsigned AddrSpace = ASOp.getImm();
      if (AddrSpace != SPIRV::StorageClass::UniformConstant) {
        if (!ST.canUseExtension(
                SPIRV::Extension::
                    SPV_EXT_relaxed_printf_string_address_space)) {
          report_fatal_error("SPV_EXT_relaxed_printf_string_address_space is "
                             "required because printf uses a format string not "
                             "in constant address space.",
                             false);
        }
        Reqs.addExtension(
            SPIRV::Extension::SPV_EXT_relaxed_printf_string_address_space);
      }
    }
  }
}

static bool isBFloat16Type(const SPIRVType *TypeDef) {
  return TypeDef && TypeDef->getNumOperands() == 3 &&
         TypeDef->getOpcode() == SPIRV::OpTypeFloat &&
         TypeDef->getOperand(1).getImm() == 16 &&
         TypeDef->getOperand(2).getImm() == SPIRV::FPEncoding::BFloat16KHR;
}

void addInstrRequirements(const MachineInstr &MI,
                          SPIRV::ModuleAnalysisInfo &MAI,
                          const SPIRVSubtarget &ST) {
  SPIRV::RequirementHandler &Reqs = MAI.Reqs;
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
  case SPIRV::OpDot: {
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    SPIRVType *TypeDef = MRI.getVRegDef(MI.getOperand(1).getReg());
    if (isBFloat16Type(TypeDef))
      Reqs.addCapability(SPIRV::Capability::BFloat16DotProductKHR);
    break;
  }
  case SPIRV::OpTypeFloat: {
    unsigned BitWidth = MI.getOperand(1).getImm();
    if (BitWidth == 64)
      Reqs.addCapability(SPIRV::Capability::Float64);
    else if (BitWidth == 16) {
      if (isBFloat16Type(&MI)) {
        if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_bfloat16))
          report_fatal_error("OpTypeFloat type with bfloat requires the "
                             "following SPIR-V extension: SPV_KHR_bfloat16",
                             false);
        Reqs.addExtension(SPIRV::Extension::SPV_KHR_bfloat16);
        Reqs.addCapability(SPIRV::Capability::BFloat16TypeKHR);
      } else {
        Reqs.addCapability(SPIRV::Capability::Float16);
      }
    }
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
    // If it's a type of pointer to float16 targeting OpenCL, add Float16Buffer
    // capability.
    if (ST.isShader())
      break;
    assert(MI.getOperand(2).isReg());
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    SPIRVType *TypeDef = MRI.getVRegDef(MI.getOperand(2).getReg());
    if ((TypeDef->getNumOperands() == 2) &&
        (TypeDef->getOpcode() == SPIRV::OpTypeFloat) &&
        (TypeDef->getOperand(1).getImm() == 16))
      Reqs.addCapability(SPIRV::Capability::Float16Buffer);
    break;
  }
  case SPIRV::OpExtInst: {
    if (MI.getOperand(2).getImm() ==
        static_cast<int64_t>(
            SPIRV::InstructionSet::NonSemantic_Shader_DebugInfo_100)) {
      Reqs.addExtension(SPIRV::Extension::SPV_KHR_non_semantic_info);
      break;
    }
    if (MI.getOperand(3).getImm() ==
        static_cast<int64_t>(SPIRV::OpenCLExtInst::printf)) {
      addPrintfRequirements(MI, Reqs, ST);
      break;
    }
    break;
  }
  case SPIRV::OpAliasDomainDeclINTEL:
  case SPIRV::OpAliasScopeDeclINTEL:
  case SPIRV::OpAliasScopeListDeclINTEL: {
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_memory_access_aliasing);
    Reqs.addCapability(SPIRV::Capability::MemoryAccessAliasingINTEL);
    break;
  }
  case SPIRV::OpBitReverse:
  case SPIRV::OpBitFieldInsert:
  case SPIRV::OpBitFieldSExtract:
  case SPIRV::OpBitFieldUExtract:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_bit_instructions)) {
      Reqs.addCapability(SPIRV::Capability::Shader);
      break;
    }
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_bit_instructions);
    Reqs.addCapability(SPIRV::Capability::BitInstructions);
    break;
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
  case SPIRV::OpInBoundsAccessChain:
  case SPIRV::OpAccessChain:
    addOpAccessChainReqs(MI, Reqs, ST);
    break;
  case SPIRV::OpTypeImage:
    addOpTypeImageReqs(MI, Reqs, ST);
    break;
  case SPIRV::OpTypeSampler:
    if (!ST.isShader()) {
      Reqs.addCapability(SPIRV::Capability::ImageBasic);
    }
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
      Reqs.addCapability(SPIRV::Capability::GroupNonUniformArithmetic);
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
  case SPIRV::OpSubgroupShuffleINTEL:
  case SPIRV::OpSubgroupShuffleDownINTEL:
  case SPIRV::OpSubgroupShuffleUpINTEL:
  case SPIRV::OpSubgroupShuffleXorINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_subgroups)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_subgroups);
      Reqs.addCapability(SPIRV::Capability::SubgroupShuffleINTEL);
    }
    break;
  case SPIRV::OpSubgroupBlockReadINTEL:
  case SPIRV::OpSubgroupBlockWriteINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_subgroups)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_subgroups);
      Reqs.addCapability(SPIRV::Capability::SubgroupBufferBlockIOINTEL);
    }
    break;
  case SPIRV::OpSubgroupImageBlockReadINTEL:
  case SPIRV::OpSubgroupImageBlockWriteINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_subgroups)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_subgroups);
      Reqs.addCapability(SPIRV::Capability::SubgroupImageBlockIOINTEL);
    }
    break;
  case SPIRV::OpSubgroupImageMediaBlockReadINTEL:
  case SPIRV::OpSubgroupImageMediaBlockWriteINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_media_block_io)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_media_block_io);
      Reqs.addCapability(SPIRV::Capability::SubgroupImageMediaBlockIOINTEL);
    }
    break;
  case SPIRV::OpAssumeTrueKHR:
  case SPIRV::OpExpectKHR:
    if (ST.canUseExtension(SPIRV::Extension::SPV_KHR_expect_assume)) {
      Reqs.addExtension(SPIRV::Extension::SPV_KHR_expect_assume);
      Reqs.addCapability(SPIRV::Capability::ExpectAssumeKHR);
    }
    break;
  case SPIRV::OpPtrCastToCrossWorkgroupINTEL:
  case SPIRV::OpCrossWorkgroupCastToPtrINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_usm_storage_classes)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_usm_storage_classes);
      Reqs.addCapability(SPIRV::Capability::USMStorageClassesINTEL);
    }
    break;
  case SPIRV::OpConstantFunctionPointerINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_function_pointers)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_function_pointers);
      Reqs.addCapability(SPIRV::Capability::FunctionPointersINTEL);
    }
    break;
  case SPIRV::OpGroupNonUniformRotateKHR:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_subgroup_rotate))
      report_fatal_error("OpGroupNonUniformRotateKHR instruction requires the "
                         "following SPIR-V extension: SPV_KHR_subgroup_rotate",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_subgroup_rotate);
    Reqs.addCapability(SPIRV::Capability::GroupNonUniformRotateKHR);
    Reqs.addCapability(SPIRV::Capability::GroupNonUniform);
    break;
  case SPIRV::OpGroupIMulKHR:
  case SPIRV::OpGroupFMulKHR:
  case SPIRV::OpGroupBitwiseAndKHR:
  case SPIRV::OpGroupBitwiseOrKHR:
  case SPIRV::OpGroupBitwiseXorKHR:
  case SPIRV::OpGroupLogicalAndKHR:
  case SPIRV::OpGroupLogicalOrKHR:
  case SPIRV::OpGroupLogicalXorKHR:
    if (ST.canUseExtension(
            SPIRV::Extension::SPV_KHR_uniform_group_instructions)) {
      Reqs.addExtension(SPIRV::Extension::SPV_KHR_uniform_group_instructions);
      Reqs.addCapability(SPIRV::Capability::GroupUniformArithmeticKHR);
    }
    break;
  case SPIRV::OpReadClockKHR:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_shader_clock))
      report_fatal_error("OpReadClockKHR instruction requires the "
                         "following SPIR-V extension: SPV_KHR_shader_clock",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_shader_clock);
    Reqs.addCapability(SPIRV::Capability::ShaderClockKHR);
    break;
  case SPIRV::OpFunctionPointerCallINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_function_pointers)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_function_pointers);
      Reqs.addCapability(SPIRV::Capability::FunctionPointersINTEL);
    }
    break;
  case SPIRV::OpAtomicFAddEXT:
  case SPIRV::OpAtomicFMinEXT:
  case SPIRV::OpAtomicFMaxEXT:
    AddAtomicFloatRequirements(MI, Reqs, ST);
    break;
  case SPIRV::OpConvertBF16ToFINTEL:
  case SPIRV::OpConvertFToBF16INTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_bfloat16_conversion)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_bfloat16_conversion);
      Reqs.addCapability(SPIRV::Capability::BFloat16ConversionINTEL);
    }
    break;
  case SPIRV::OpRoundFToTF32INTEL:
    if (ST.canUseExtension(
            SPIRV::Extension::SPV_INTEL_tensor_float32_conversion)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_tensor_float32_conversion);
      Reqs.addCapability(SPIRV::Capability::TensorFloat32RoundingINTEL);
    }
    break;
  case SPIRV::OpVariableLengthArrayINTEL:
  case SPIRV::OpSaveMemoryINTEL:
  case SPIRV::OpRestoreMemoryINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_variable_length_array)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_variable_length_array);
      Reqs.addCapability(SPIRV::Capability::VariableLengthArrayINTEL);
    }
    break;
  case SPIRV::OpAsmTargetINTEL:
  case SPIRV::OpAsmINTEL:
  case SPIRV::OpAsmCallINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_inline_assembly)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_inline_assembly);
      Reqs.addCapability(SPIRV::Capability::AsmINTEL);
    }
    break;
  case SPIRV::OpTypeCooperativeMatrixKHR: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix))
      report_fatal_error(
          "OpTypeCooperativeMatrixKHR type requires the "
          "following SPIR-V extension: SPV_KHR_cooperative_matrix",
          false);
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix);
    Reqs.addCapability(SPIRV::Capability::CooperativeMatrixKHR);
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    SPIRVType *TypeDef = MRI.getVRegDef(MI.getOperand(1).getReg());
    if (isBFloat16Type(TypeDef))
      Reqs.addCapability(SPIRV::Capability::BFloat16CooperativeMatrixKHR);
    break;
  }
  case SPIRV::OpArithmeticFenceEXT:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_EXT_arithmetic_fence))
      report_fatal_error("OpArithmeticFenceEXT requires the "
                         "following SPIR-V extension: SPV_EXT_arithmetic_fence",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_EXT_arithmetic_fence);
    Reqs.addCapability(SPIRV::Capability::ArithmeticFenceEXT);
    break;
  case SPIRV::OpControlBarrierArriveINTEL:
  case SPIRV::OpControlBarrierWaitINTEL:
    if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_split_barrier)) {
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_split_barrier);
      Reqs.addCapability(SPIRV::Capability::SplitBarrierINTEL);
    }
    break;
  case SPIRV::OpCooperativeMatrixMulAddKHR: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix))
      report_fatal_error("Cooperative matrix instructions require the "
                         "following SPIR-V extension: "
                         "SPV_KHR_cooperative_matrix",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix);
    Reqs.addCapability(SPIRV::Capability::CooperativeMatrixKHR);
    constexpr unsigned MulAddMaxSize = 6;
    if (MI.getNumOperands() != MulAddMaxSize)
      break;
    const int64_t CoopOperands = MI.getOperand(MulAddMaxSize - 1).getImm();
    if (CoopOperands &
        SPIRV::CooperativeMatrixOperands::MatrixAAndBTF32ComponentsINTEL) {
      if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix))
        report_fatal_error("MatrixAAndBTF32ComponentsINTEL type interpretation "
                           "require the following SPIR-V extension: "
                           "SPV_INTEL_joint_matrix",
                           false);
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
      Reqs.addCapability(
          SPIRV::Capability::CooperativeMatrixTF32ComponentTypeINTEL);
    }
    if (CoopOperands & SPIRV::CooperativeMatrixOperands::
                           MatrixAAndBBFloat16ComponentsINTEL ||
        CoopOperands &
            SPIRV::CooperativeMatrixOperands::MatrixCBFloat16ComponentsINTEL ||
        CoopOperands & SPIRV::CooperativeMatrixOperands::
                           MatrixResultBFloat16ComponentsINTEL) {
      if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix))
        report_fatal_error("***BF16ComponentsINTEL type interpretations "
                           "require the following SPIR-V extension: "
                           "SPV_INTEL_joint_matrix",
                           false);
      Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
      Reqs.addCapability(
          SPIRV::Capability::CooperativeMatrixBFloat16ComponentTypeINTEL);
    }
    break;
  }
  case SPIRV::OpCooperativeMatrixLoadKHR:
  case SPIRV::OpCooperativeMatrixStoreKHR:
  case SPIRV::OpCooperativeMatrixLoadCheckedINTEL:
  case SPIRV::OpCooperativeMatrixStoreCheckedINTEL:
  case SPIRV::OpCooperativeMatrixPrefetchINTEL: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix))
      report_fatal_error("Cooperative matrix instructions require the "
                         "following SPIR-V extension: "
                         "SPV_KHR_cooperative_matrix",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_KHR_cooperative_matrix);
    Reqs.addCapability(SPIRV::Capability::CooperativeMatrixKHR);

    // Check Layout operand in case if it's not a standard one and add the
    // appropriate capability.
    std::unordered_map<unsigned, unsigned> LayoutToInstMap = {
        {SPIRV::OpCooperativeMatrixLoadKHR, 3},
        {SPIRV::OpCooperativeMatrixStoreKHR, 2},
        {SPIRV::OpCooperativeMatrixLoadCheckedINTEL, 5},
        {SPIRV::OpCooperativeMatrixStoreCheckedINTEL, 4},
        {SPIRV::OpCooperativeMatrixPrefetchINTEL, 4}};

    const auto OpCode = MI.getOpcode();
    const unsigned LayoutNum = LayoutToInstMap[OpCode];
    Register RegLayout = MI.getOperand(LayoutNum).getReg();
    const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
    MachineInstr *MILayout = MRI.getUniqueVRegDef(RegLayout);
    if (MILayout->getOpcode() == SPIRV::OpConstantI) {
      const unsigned LayoutVal = MILayout->getOperand(2).getImm();
      if (LayoutVal ==
          static_cast<unsigned>(SPIRV::CooperativeMatrixLayout::PackedINTEL)) {
        if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix))
          report_fatal_error("PackedINTEL layout require the following SPIR-V "
                             "extension: SPV_INTEL_joint_matrix",
                             false);
        Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
        Reqs.addCapability(SPIRV::Capability::PackedCooperativeMatrixINTEL);
      }
    }

    // Nothing to do.
    if (OpCode == SPIRV::OpCooperativeMatrixLoadKHR ||
        OpCode == SPIRV::OpCooperativeMatrixStoreKHR)
      break;

    std::string InstName;
    switch (OpCode) {
    case SPIRV::OpCooperativeMatrixPrefetchINTEL:
      InstName = "OpCooperativeMatrixPrefetchINTEL";
      break;
    case SPIRV::OpCooperativeMatrixLoadCheckedINTEL:
      InstName = "OpCooperativeMatrixLoadCheckedINTEL";
      break;
    case SPIRV::OpCooperativeMatrixStoreCheckedINTEL:
      InstName = "OpCooperativeMatrixStoreCheckedINTEL";
      break;
    }

    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix)) {
      const std::string ErrorMsg =
          InstName + " instruction requires the "
                     "following SPIR-V extension: SPV_INTEL_joint_matrix";
      report_fatal_error(ErrorMsg.c_str(), false);
    }
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
    if (OpCode == SPIRV::OpCooperativeMatrixPrefetchINTEL) {
      Reqs.addCapability(SPIRV::Capability::CooperativeMatrixPrefetchINTEL);
      break;
    }
    Reqs.addCapability(
        SPIRV::Capability::CooperativeMatrixCheckedInstructionsINTEL);
    break;
  }
  case SPIRV::OpCooperativeMatrixConstructCheckedINTEL:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix))
      report_fatal_error("OpCooperativeMatrixConstructCheckedINTEL "
                         "instructions require the following SPIR-V extension: "
                         "SPV_INTEL_joint_matrix",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
    Reqs.addCapability(
        SPIRV::Capability::CooperativeMatrixCheckedInstructionsINTEL);
    break;
  case SPIRV::OpCooperativeMatrixGetElementCoordINTEL:
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_joint_matrix))
      report_fatal_error("OpCooperativeMatrixGetElementCoordINTEL requires the "
                         "following SPIR-V extension: SPV_INTEL_joint_matrix",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_joint_matrix);
    Reqs.addCapability(
        SPIRV::Capability::CooperativeMatrixInvocationInstructionsINTEL);
    break;
  case SPIRV::OpConvertHandleToImageINTEL:
  case SPIRV::OpConvertHandleToSamplerINTEL:
  case SPIRV::OpConvertHandleToSampledImageINTEL: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_bindless_images))
      report_fatal_error("OpConvertHandleTo[Image/Sampler/SampledImage]INTEL "
                         "instructions require the following SPIR-V extension: "
                         "SPV_INTEL_bindless_images",
                         false);
    SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
    SPIRV::AddressingModel::AddressingModel AddrModel = MAI.Addr;
    SPIRVType *TyDef = GR->getSPIRVTypeForVReg(MI.getOperand(1).getReg());
    if (MI.getOpcode() == SPIRV::OpConvertHandleToImageINTEL &&
        TyDef->getOpcode() != SPIRV::OpTypeImage) {
      report_fatal_error("Incorrect return type for the instruction "
                         "OpConvertHandleToImageINTEL",
                         false);
    } else if (MI.getOpcode() == SPIRV::OpConvertHandleToSamplerINTEL &&
               TyDef->getOpcode() != SPIRV::OpTypeSampler) {
      report_fatal_error("Incorrect return type for the instruction "
                         "OpConvertHandleToSamplerINTEL",
                         false);
    } else if (MI.getOpcode() == SPIRV::OpConvertHandleToSampledImageINTEL &&
               TyDef->getOpcode() != SPIRV::OpTypeSampledImage) {
      report_fatal_error("Incorrect return type for the instruction "
                         "OpConvertHandleToSampledImageINTEL",
                         false);
    }
    SPIRVType *SpvTy = GR->getSPIRVTypeForVReg(MI.getOperand(2).getReg());
    unsigned Bitwidth = GR->getScalarOrVectorBitWidth(SpvTy);
    if (!(Bitwidth == 32 && AddrModel == SPIRV::AddressingModel::Physical32) &&
        !(Bitwidth == 64 && AddrModel == SPIRV::AddressingModel::Physical64)) {
      report_fatal_error(
          "Parameter value must be a 32-bit scalar in case of "
          "Physical32 addressing model or a 64-bit scalar in case of "
          "Physical64 addressing model",
          false);
    }
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_bindless_images);
    Reqs.addCapability(SPIRV::Capability::BindlessImagesINTEL);
    break;
  }
  case SPIRV::OpSubgroup2DBlockLoadINTEL:
  case SPIRV::OpSubgroup2DBlockLoadTransposeINTEL:
  case SPIRV::OpSubgroup2DBlockLoadTransformINTEL:
  case SPIRV::OpSubgroup2DBlockPrefetchINTEL:
  case SPIRV::OpSubgroup2DBlockStoreINTEL: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_2d_block_io))
      report_fatal_error("OpSubgroup2DBlock[Load/LoadTranspose/LoadTransform/"
                         "Prefetch/Store]INTEL instructions require the "
                         "following SPIR-V extension: SPV_INTEL_2d_block_io",
                         false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_2d_block_io);
    Reqs.addCapability(SPIRV::Capability::Subgroup2DBlockIOINTEL);

    const auto OpCode = MI.getOpcode();
    if (OpCode == SPIRV::OpSubgroup2DBlockLoadTransposeINTEL) {
      Reqs.addCapability(SPIRV::Capability::Subgroup2DBlockTransposeINTEL);
      break;
    }
    if (OpCode == SPIRV::OpSubgroup2DBlockLoadTransformINTEL) {
      Reqs.addCapability(SPIRV::Capability::Subgroup2DBlockTransformINTEL);
      break;
    }
    break;
  }
  case SPIRV::OpKill: {
    Reqs.addCapability(SPIRV::Capability::Shader);
  } break;
  case SPIRV::OpDemoteToHelperInvocation:
    Reqs.addCapability(SPIRV::Capability::DemoteToHelperInvocation);

    if (ST.canUseExtension(
            SPIRV::Extension::SPV_EXT_demote_to_helper_invocation)) {
      if (!ST.isAtLeastSPIRVVer(llvm::VersionTuple(1, 6)))
        Reqs.addExtension(
            SPIRV::Extension::SPV_EXT_demote_to_helper_invocation);
    }
    break;
  case SPIRV::OpSDot:
  case SPIRV::OpUDot:
  case SPIRV::OpSUDot:
  case SPIRV::OpSDotAccSat:
  case SPIRV::OpUDotAccSat:
  case SPIRV::OpSUDotAccSat:
    AddDotProductRequirements(MI, Reqs, ST);
    break;
  case SPIRV::OpImageRead: {
    Register ImageReg = MI.getOperand(2).getReg();
    SPIRVType *TypeDef = ST.getSPIRVGlobalRegistry()->getResultType(
        ImageReg, const_cast<MachineFunction *>(MI.getMF()));
    // OpImageRead and OpImageWrite can use Unknown Image Formats
    // when the Kernel capability is declared. In the OpenCL environment we are
    // not allowed to produce
    // StorageImageReadWithoutFormat/StorageImageWriteWithoutFormat, see
    // https://github.com/KhronosGroup/SPIRV-Headers/issues/487

    if (isImageTypeWithUnknownFormat(TypeDef) && ST.isShader())
      Reqs.addCapability(SPIRV::Capability::StorageImageReadWithoutFormat);
    break;
  }
  case SPIRV::OpImageWrite: {
    Register ImageReg = MI.getOperand(0).getReg();
    SPIRVType *TypeDef = ST.getSPIRVGlobalRegistry()->getResultType(
        ImageReg, const_cast<MachineFunction *>(MI.getMF()));
    // OpImageRead and OpImageWrite can use Unknown Image Formats
    // when the Kernel capability is declared. In the OpenCL environment we are
    // not allowed to produce
    // StorageImageReadWithoutFormat/StorageImageWriteWithoutFormat, see
    // https://github.com/KhronosGroup/SPIRV-Headers/issues/487

    if (isImageTypeWithUnknownFormat(TypeDef) && ST.isShader())
      Reqs.addCapability(SPIRV::Capability::StorageImageWriteWithoutFormat);
    break;
  }
  case SPIRV::OpTypeStructContinuedINTEL:
  case SPIRV::OpConstantCompositeContinuedINTEL:
  case SPIRV::OpSpecConstantCompositeContinuedINTEL:
  case SPIRV::OpCompositeConstructContinuedINTEL: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_long_composites))
      report_fatal_error(
          "Continued instructions require the "
          "following SPIR-V extension: SPV_INTEL_long_composites",
          false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_long_composites);
    Reqs.addCapability(SPIRV::Capability::LongCompositesINTEL);
    break;
  }
  case SPIRV::OpSubgroupMatrixMultiplyAccumulateINTEL: {
    if (!ST.canUseExtension(
            SPIRV::Extension::SPV_INTEL_subgroup_matrix_multiply_accumulate))
      report_fatal_error(
          "OpSubgroupMatrixMultiplyAccumulateINTEL instruction requires the "
          "following SPIR-V "
          "extension: SPV_INTEL_subgroup_matrix_multiply_accumulate",
          false);
    Reqs.addExtension(
        SPIRV::Extension::SPV_INTEL_subgroup_matrix_multiply_accumulate);
    Reqs.addCapability(
        SPIRV::Capability::SubgroupMatrixMultiplyAccumulateINTEL);
    break;
  }
  case SPIRV::OpBitwiseFunctionINTEL: {
    if (!ST.canUseExtension(
            SPIRV::Extension::SPV_INTEL_ternary_bitwise_function))
      report_fatal_error(
          "OpBitwiseFunctionINTEL instruction requires the following SPIR-V "
          "extension: SPV_INTEL_ternary_bitwise_function",
          false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_ternary_bitwise_function);
    Reqs.addCapability(SPIRV::Capability::TernaryBitwiseFunctionINTEL);
    break;
  }
  case SPIRV::OpCopyMemorySized: {
    Reqs.addCapability(SPIRV::Capability::Addresses);
    // TODO: Add UntypedPointersKHR when implemented.
    break;
  }
  case SPIRV::OpPredicatedLoadINTEL:
  case SPIRV::OpPredicatedStoreINTEL: {
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_predicated_io))
      report_fatal_error(
          "OpPredicated[Load/Store]INTEL instructions require "
          "the following SPIR-V extension: SPV_INTEL_predicated_io",
          false);
    Reqs.addExtension(SPIRV::Extension::SPV_INTEL_predicated_io);
    Reqs.addCapability(SPIRV::Capability::PredicatedIOINTEL);
    break;
  }

  default:
    break;
  }

  // If we require capability Shader, then we can remove the requirement for
  // the BitInstructions capability, since Shader is a superset capability
  // of BitInstructions.
  Reqs.removeCapabilityIf(SPIRV::Capability::BitInstructions,
                          SPIRV::Capability::Shader);
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
        addInstrRequirements(MI, MAI, ST);
  }
  // Collect requirements for OpExecutionMode instructions.
  auto Node = M.getNamedMetadata("spirv.ExecutionMode");
  if (Node) {
    bool RequireFloatControls = false, RequireIntelFloatControls2 = false,
         RequireKHRFloatControls2 = false,
         VerLower14 = !ST.isAtLeastSPIRVVer(VersionTuple(1, 4));
    bool HasIntelFloatControls2 =
        ST.canUseExtension(SPIRV::Extension::SPV_INTEL_float_controls2);
    bool HasKHRFloatControls2 =
        ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2);
    for (unsigned i = 0; i < Node->getNumOperands(); i++) {
      MDNode *MDN = cast<MDNode>(Node->getOperand(i));
      const MDOperand &MDOp = MDN->getOperand(1);
      if (auto *CMeta = dyn_cast<ConstantAsMetadata>(MDOp)) {
        Constant *C = CMeta->getValue();
        if (ConstantInt *Const = dyn_cast<ConstantInt>(C)) {
          auto EM = Const->getZExtValue();
          // SPV_KHR_float_controls is not available until v1.4:
          // add SPV_KHR_float_controls if the version is too low
          switch (EM) {
          case SPIRV::ExecutionMode::DenormPreserve:
          case SPIRV::ExecutionMode::DenormFlushToZero:
          case SPIRV::ExecutionMode::RoundingModeRTE:
          case SPIRV::ExecutionMode::RoundingModeRTZ:
            RequireFloatControls = VerLower14;
            MAI.Reqs.getAndAddRequirements(
                SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
            break;
          case SPIRV::ExecutionMode::RoundingModeRTPINTEL:
          case SPIRV::ExecutionMode::RoundingModeRTNINTEL:
          case SPIRV::ExecutionMode::FloatingPointModeALTINTEL:
          case SPIRV::ExecutionMode::FloatingPointModeIEEEINTEL:
            if (HasIntelFloatControls2) {
              RequireIntelFloatControls2 = true;
              MAI.Reqs.getAndAddRequirements(
                  SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
            }
            break;
          case SPIRV::ExecutionMode::FPFastMathDefault: {
            if (HasKHRFloatControls2) {
              RequireKHRFloatControls2 = true;
              MAI.Reqs.getAndAddRequirements(
                  SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
            }
            break;
          }
          case SPIRV::ExecutionMode::ContractionOff:
          case SPIRV::ExecutionMode::SignedZeroInfNanPreserve:
            if (HasKHRFloatControls2) {
              RequireKHRFloatControls2 = true;
              MAI.Reqs.getAndAddRequirements(
                  SPIRV::OperandCategory::ExecutionModeOperand,
                  SPIRV::ExecutionMode::FPFastMathDefault, ST);
            } else {
              MAI.Reqs.getAndAddRequirements(
                  SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
            }
            break;
          default:
            MAI.Reqs.getAndAddRequirements(
                SPIRV::OperandCategory::ExecutionModeOperand, EM, ST);
          }
        }
      }
    }
    if (RequireFloatControls &&
        ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls))
      MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_float_controls);
    if (RequireIntelFloatControls2)
      MAI.Reqs.addExtension(SPIRV::Extension::SPV_INTEL_float_controls2);
    if (RequireKHRFloatControls2)
      MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_float_controls2);
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
    if (F.getFnAttribute("enable-maximal-reconvergence").getValueAsBool()) {
      MAI.Reqs.addExtension(SPIRV::Extension::SPV_KHR_maximal_reconvergence);
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

    if (F.hasOptNone()) {
      if (ST.canUseExtension(SPIRV::Extension::SPV_INTEL_optnone)) {
        MAI.Reqs.addExtension(SPIRV::Extension::SPV_INTEL_optnone);
        MAI.Reqs.addCapability(SPIRV::Capability::OptNoneINTEL);
      } else if (ST.canUseExtension(SPIRV::Extension::SPV_EXT_optnone)) {
        MAI.Reqs.addExtension(SPIRV::Extension::SPV_EXT_optnone);
        MAI.Reqs.addCapability(SPIRV::Capability::OptNoneEXT);
      }
    }
  }
}

static unsigned getFastMathFlags(const MachineInstr &I,
                                 const SPIRVSubtarget &ST) {
  unsigned Flags = SPIRV::FPFastMathMode::None;
  bool CanUseKHRFloatControls2 =
      ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2);
  if (I.getFlag(MachineInstr::MIFlag::FmNoNans))
    Flags |= SPIRV::FPFastMathMode::NotNaN;
  if (I.getFlag(MachineInstr::MIFlag::FmNoInfs))
    Flags |= SPIRV::FPFastMathMode::NotInf;
  if (I.getFlag(MachineInstr::MIFlag::FmNsz))
    Flags |= SPIRV::FPFastMathMode::NSZ;
  if (I.getFlag(MachineInstr::MIFlag::FmArcp))
    Flags |= SPIRV::FPFastMathMode::AllowRecip;
  if (I.getFlag(MachineInstr::MIFlag::FmContract) && CanUseKHRFloatControls2)
    Flags |= SPIRV::FPFastMathMode::AllowContract;
  if (I.getFlag(MachineInstr::MIFlag::FmReassoc)) {
    if (CanUseKHRFloatControls2)
      // LLVM reassoc maps to SPIRV transform, see
      // https://github.com/KhronosGroup/SPIRV-Registry/issues/326 for details.
      // Because we are enabling AllowTransform, we must enable AllowReassoc and
      // AllowContract too, as required by SPIRV spec. Also, we used to map
      // MIFlag::FmReassoc to FPFastMathMode::Fast, which now should instead by
      // replaced by turning all the other bits instead. Therefore, we're
      // enabling every bit here except None and Fast.
      Flags |= SPIRV::FPFastMathMode::NotNaN | SPIRV::FPFastMathMode::NotInf |
               SPIRV::FPFastMathMode::NSZ | SPIRV::FPFastMathMode::AllowRecip |
               SPIRV::FPFastMathMode::AllowTransform |
               SPIRV::FPFastMathMode::AllowReassoc |
               SPIRV::FPFastMathMode::AllowContract;
    else
      Flags |= SPIRV::FPFastMathMode::Fast;
  }

  if (CanUseKHRFloatControls2) {
    // Error out if SPIRV::FPFastMathMode::Fast is enabled.
    assert(!(Flags & SPIRV::FPFastMathMode::Fast) &&
           "SPIRV::FPFastMathMode::Fast is deprecated and should not be used "
           "anymore.");

    // Error out if AllowTransform is enabled without AllowReassoc and
    // AllowContract.
    assert((!(Flags & SPIRV::FPFastMathMode::AllowTransform) ||
            ((Flags & SPIRV::FPFastMathMode::AllowReassoc &&
              Flags & SPIRV::FPFastMathMode::AllowContract))) &&
           "SPIRV::FPFastMathMode::AllowTransform requires AllowReassoc and "
           "AllowContract flags to be enabled as well.");
  }

  return Flags;
}

static bool isFastMathModeAvailable(const SPIRVSubtarget &ST) {
  if (ST.isKernel())
    return true;
  if (ST.getSPIRVVersion() < VersionTuple(1, 2))
    return false;
  return ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2);
}

static void handleMIFlagDecoration(
    MachineInstr &I, const SPIRVSubtarget &ST, const SPIRVInstrInfo &TII,
    SPIRV::RequirementHandler &Reqs, const SPIRVGlobalRegistry *GR,
    SPIRV::FPFastMathDefaultInfoVector &FPFastMathDefaultInfoVec) {
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
  if (!TII.canUseFastMathFlags(
          I, ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2)))
    return;

  unsigned FMFlags = getFastMathFlags(I, ST);
  if (FMFlags == SPIRV::FPFastMathMode::None) {
    // We also need to check if any FPFastMathDefault info was set for the
    // types used in this instruction.
    if (FPFastMathDefaultInfoVec.empty())
      return;

    // There are three types of instructions that can use fast math flags:
    // 1. Arithmetic instructions (FAdd, FMul, FSub, FDiv, FRem, etc.)
    // 2. Relational instructions (FCmp, FOrd, FUnord, etc.)
    // 3. Extended instructions (ExtInst)
    // For arithmetic instructions, the floating point type can be in the
    // result type or in the operands, but they all must be the same.
    // For the relational and logical instructions, the floating point type
    // can only be in the operands 1 and 2, not the result type. Also, the
    // operands must have the same type. For the extended instructions, the
    // floating point type can be in the result type or in the operands. It's
    // unclear if the operands and the result type must be the same. Let's
    // assume they must be. Therefore, for 1. and 2., we can check the first
    // operand type, and for 3. we can check the result type.
    assert(I.getNumOperands() >= 3 && "Expected at least 3 operands");
    Register ResReg = I.getOpcode() == SPIRV::OpExtInst
                          ? I.getOperand(1).getReg()
                          : I.getOperand(2).getReg();
    SPIRVType *ResType = GR->getSPIRVTypeForVReg(ResReg, I.getMF());
    const Type *Ty = GR->getTypeForSPIRVType(ResType);
    Ty = Ty->isVectorTy() ? cast<VectorType>(Ty)->getElementType() : Ty;

    // Match instruction type with the FPFastMathDefaultInfoVec.
    bool Emit = false;
    for (SPIRV::FPFastMathDefaultInfo &Elem : FPFastMathDefaultInfoVec) {
      if (Ty == Elem.Ty) {
        FMFlags = Elem.FastMathFlags;
        Emit = Elem.ContractionOff || Elem.SignedZeroInfNanPreserve ||
               Elem.FPFastMathDefault;
        break;
      }
    }

    if (FMFlags == SPIRV::FPFastMathMode::None && !Emit)
      return;
  }
  if (isFastMathModeAvailable(ST)) {
    Register DstReg = I.getOperand(0).getReg();
    buildOpDecorate(DstReg, I, TII, SPIRV::Decoration::FPFastMathMode,
                    {FMFlags});
  }
}

// Walk all functions and add decorations related to MI flags.
static void addDecorations(const Module &M, const SPIRVInstrInfo &TII,
                           MachineModuleInfo *MMI, const SPIRVSubtarget &ST,
                           SPIRV::ModuleAnalysisInfo &MAI,
                           const SPIRVGlobalRegistry *GR) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;

    for (auto &MBB : *MF)
      for (auto &MI : MBB)
        handleMIFlagDecoration(MI, ST, TII, MAI.Reqs, GR,
                               MAI.FPFastMathDefaultInfoMap[&(*F)]);
  }
}

static void addMBBNames(const Module &M, const SPIRVInstrInfo &TII,
                        MachineModuleInfo *MMI, const SPIRVSubtarget &ST,
                        SPIRV::ModuleAnalysisInfo &MAI) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    MachineRegisterInfo &MRI = MF->getRegInfo();
    for (auto &MBB : *MF) {
      if (!MBB.hasName() || MBB.empty())
        continue;
      // Emit basic block names.
      Register Reg = MRI.createGenericVirtualRegister(LLT::scalar(64));
      MRI.setRegClass(Reg, &SPIRV::IDRegClass);
      buildOpName(Reg, MBB.getName(), *std::prev(MBB.end()), TII);
      MCRegister GlobalReg = MAI.getOrCreateMBBRegister(MBB);
      MAI.setRegisterAlias(MF, Reg, GlobalReg);
    }
  }
}

// patching Instruction::PHI to SPIRV::OpPhi
static void patchPhis(const Module &M, SPIRVGlobalRegistry *GR,
                      const SPIRVInstrInfo &TII, MachineModuleInfo *MMI) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    MachineFunction *MF = MMI->getMachineFunction(*F);
    if (!MF)
      continue;
    for (auto &MBB : *MF) {
      for (MachineInstr &MI : MBB.phis()) {
        MI.setDesc(TII.get(SPIRV::OpPhi));
        Register ResTypeReg = GR->getSPIRVTypeID(
            GR->getSPIRVTypeForVReg(MI.getOperand(0).getReg(), MF));
        MI.insert(MI.operands_begin() + 1,
                  {MachineOperand::CreateReg(ResTypeReg, false)});
      }
    }

    MF->getProperties().setNoPHIs();
  }
}

static SPIRV::FPFastMathDefaultInfoVector &getOrCreateFPFastMathDefaultInfoVec(
    const Module &M, SPIRV::ModuleAnalysisInfo &MAI, const Function *F) {
  auto it = MAI.FPFastMathDefaultInfoMap.find(F);
  if (it != MAI.FPFastMathDefaultInfoMap.end())
    return it->second;

  // If the map does not contain the entry, create a new one. Initialize it to
  // contain all 3 elements sorted by bit width of target type: {half, float,
  // double}.
  SPIRV::FPFastMathDefaultInfoVector FPFastMathDefaultInfoVec;
  FPFastMathDefaultInfoVec.emplace_back(Type::getHalfTy(M.getContext()),
                                        SPIRV::FPFastMathMode::None);
  FPFastMathDefaultInfoVec.emplace_back(Type::getFloatTy(M.getContext()),
                                        SPIRV::FPFastMathMode::None);
  FPFastMathDefaultInfoVec.emplace_back(Type::getDoubleTy(M.getContext()),
                                        SPIRV::FPFastMathMode::None);
  return MAI.FPFastMathDefaultInfoMap[F] = std::move(FPFastMathDefaultInfoVec);
}

static SPIRV::FPFastMathDefaultInfo &getFPFastMathDefaultInfo(
    SPIRV::FPFastMathDefaultInfoVector &FPFastMathDefaultInfoVec,
    const Type *Ty) {
  size_t BitWidth = Ty->getScalarSizeInBits();
  int Index =
      SPIRV::FPFastMathDefaultInfoVector::computeFPFastMathDefaultInfoVecIndex(
          BitWidth);
  assert(Index >= 0 && Index < 3 &&
         "Expected FPFastMathDefaultInfo for half, float, or double");
  assert(FPFastMathDefaultInfoVec.size() == 3 &&
         "Expected FPFastMathDefaultInfoVec to have exactly 3 elements");
  return FPFastMathDefaultInfoVec[Index];
}

static void collectFPFastMathDefaults(const Module &M,
                                      SPIRV::ModuleAnalysisInfo &MAI,
                                      const SPIRVSubtarget &ST) {
  if (!ST.canUseExtension(SPIRV::Extension::SPV_KHR_float_controls2))
    return;

  // Store the FPFastMathDefaultInfo in the FPFastMathDefaultInfoMap.
  // We need the entry point (function) as the key, and the target
  // type and flags as the value.
  // We also need to check ContractionOff and SignedZeroInfNanPreserve
  // execution modes, as they are now deprecated and must be replaced
  // with FPFastMathDefaultInfo.
  auto Node = M.getNamedMetadata("spirv.ExecutionMode");
  if (!Node)
    return;

  for (unsigned i = 0; i < Node->getNumOperands(); i++) {
    MDNode *MDN = cast<MDNode>(Node->getOperand(i));
    assert(MDN->getNumOperands() >= 2 && "Expected at least 2 operands");
    const Function *F = cast<Function>(
        cast<ConstantAsMetadata>(MDN->getOperand(0))->getValue());
    const auto EM =
        cast<ConstantInt>(
            cast<ConstantAsMetadata>(MDN->getOperand(1))->getValue())
            ->getZExtValue();
    if (EM == SPIRV::ExecutionMode::FPFastMathDefault) {
      assert(MDN->getNumOperands() == 4 &&
             "Expected 4 operands for FPFastMathDefault");

      const Type *T = cast<ValueAsMetadata>(MDN->getOperand(2))->getType();
      unsigned Flags =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(MDN->getOperand(3))->getValue())
              ->getZExtValue();
      SPIRV::FPFastMathDefaultInfoVector &FPFastMathDefaultInfoVec =
          getOrCreateFPFastMathDefaultInfoVec(M, MAI, F);
      SPIRV::FPFastMathDefaultInfo &Info =
          getFPFastMathDefaultInfo(FPFastMathDefaultInfoVec, T);
      Info.FastMathFlags = Flags;
      Info.FPFastMathDefault = true;
    } else if (EM == SPIRV::ExecutionMode::ContractionOff) {
      assert(MDN->getNumOperands() == 2 &&
             "Expected no operands for ContractionOff");

      // We need to save this info for every possible FP type, i.e. {half,
      // float, double, fp128}.
      SPIRV::FPFastMathDefaultInfoVector &FPFastMathDefaultInfoVec =
          getOrCreateFPFastMathDefaultInfoVec(M, MAI, F);
      for (SPIRV::FPFastMathDefaultInfo &Info : FPFastMathDefaultInfoVec) {
        Info.ContractionOff = true;
      }
    } else if (EM == SPIRV::ExecutionMode::SignedZeroInfNanPreserve) {
      assert(MDN->getNumOperands() == 3 &&
             "Expected 1 operand for SignedZeroInfNanPreserve");
      unsigned TargetWidth =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(MDN->getOperand(2))->getValue())
              ->getZExtValue();
      // We need to save this info only for the FP type with TargetWidth.
      SPIRV::FPFastMathDefaultInfoVector &FPFastMathDefaultInfoVec =
          getOrCreateFPFastMathDefaultInfoVec(M, MAI, F);
      int Index = SPIRV::FPFastMathDefaultInfoVector::
          computeFPFastMathDefaultInfoVecIndex(TargetWidth);
      assert(Index >= 0 && Index < 3 &&
             "Expected FPFastMathDefaultInfo for half, float, or double");
      assert(FPFastMathDefaultInfoVec.size() == 3 &&
             "Expected FPFastMathDefaultInfoVec to have exactly 3 elements");
      FPFastMathDefaultInfoVec[Index].SignedZeroInfNanPreserve = true;
    }
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

  patchPhis(M, GR, *TII, MMI);

  addMBBNames(M, *TII, MMI, *ST, MAI);
  collectFPFastMathDefaults(M, MAI, *ST);
  addDecorations(M, *TII, MMI, *ST, MAI, GR);

  collectReqs(M, MAI, MMI, *ST);

  // Process type/const/global var/func decl instructions, number their
  // destination registers from 0 to N, collect Extensions and Capabilities.
  collectReqs(M, MAI, MMI, *ST);
  collectDeclarations(M);

  // Number rest of registers from N+1 onwards.
  numberRegistersGlobally(M);

  // Collect OpName, OpEntryPoint, OpDecorate etc, process other instructions.
  processOtherInstrs(M);

  // If there are no entry points, we need the Linkage capability.
  if (MAI.MS[SPIRV::MB_EntryPoints].empty())
    MAI.Reqs.addCapability(SPIRV::Capability::Linkage);

  // Set maximum ID used.
  GR->setBound(MAI.MaxID);

  return false;
}
