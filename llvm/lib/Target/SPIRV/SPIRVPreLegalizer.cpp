//===-- SPIRVPreLegalizer.cpp - prepare IR for legalization -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass prepares IR for legalization: it assigns SPIR-V types to registers
// and removes intrinsics which holded these types during IR translation.
// Also it processes constants and registers them in GR to avoid duplication.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#define DEBUG_TYPE "spirv-prelegalizer"

using namespace llvm;

namespace {
class SPIRVPreLegalizer : public MachineFunctionPass {
public:
  static char ID;
  SPIRVPreLegalizer() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // namespace

void SPIRVPreLegalizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static void
addConstantsToTrack(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                    const SPIRVSubtarget &STI,
                    DenseMap<MachineInstr *, Type *> &TargetExtConstTypes) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DenseMap<MachineInstr *, Register> RegsAlreadyAddedToDT;
  SmallVector<MachineInstr *, 10> ToErase, ToEraseComposites;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_track_constant))
        continue;
      ToErase.push_back(&MI);
      Register SrcReg = MI.getOperand(2).getReg();
      auto *Const =
          cast<Constant>(cast<ConstantAsMetadata>(
                             MI.getOperand(3).getMetadata()->getOperand(0))
                             ->getValue());
      if (auto *GV = dyn_cast<GlobalValue>(Const)) {
        Register Reg = GR->find(GV, &MF);
        if (!Reg.isValid()) {
          GR->add(GV, MRI.getVRegDef(SrcReg));
          GR->addGlobalObject(GV, &MF, SrcReg);
        } else
          RegsAlreadyAddedToDT[&MI] = Reg;
      } else {
        Register Reg = GR->find(Const, &MF);
        if (!Reg.isValid()) {
          if (auto *ConstVec = dyn_cast<ConstantDataVector>(Const)) {
            auto *BuildVec = MRI.getVRegDef(SrcReg);
            assert(BuildVec &&
                   BuildVec->getOpcode() == TargetOpcode::G_BUILD_VECTOR);
            GR->add(Const, BuildVec);
            for (unsigned i = 0; i < ConstVec->getNumElements(); ++i) {
              // Ensure that OpConstantComposite reuses a constant when it's
              // already created and available in the same machine function.
              Constant *ElemConst = ConstVec->getElementAsConstant(i);
              Register ElemReg = GR->find(ElemConst, &MF);
              if (!ElemReg.isValid())
                GR->add(ElemConst,
                        MRI.getVRegDef(BuildVec->getOperand(1 + i).getReg()));
              else
                BuildVec->getOperand(1 + i).setReg(ElemReg);
            }
          }
          if (Const->getType()->isTargetExtTy()) {
            // remember association so that we can restore it when assign types
            MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
            if (SrcMI)
              GR->add(Const, SrcMI);
            if (SrcMI && (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT ||
                          SrcMI->getOpcode() == TargetOpcode::G_IMPLICIT_DEF))
              TargetExtConstTypes[SrcMI] = Const->getType();
            if (Const->isNullValue()) {
              MachineBasicBlock &DepMBB = MF.front();
              MachineIRBuilder MIB(DepMBB, DepMBB.getFirstNonPHI());
              SPIRVType *ExtType = GR->getOrCreateSPIRVType(
                  Const->getType(), MIB, SPIRV::AccessQualifier::ReadWrite,
                  true);
              assert(SrcMI && "Expected source instruction to be valid");
              SrcMI->setDesc(STI.getInstrInfo()->get(SPIRV::OpConstantNull));
              SrcMI->addOperand(MachineOperand::CreateReg(
                  GR->getSPIRVTypeID(ExtType), false));
            }
          }
        } else {
          RegsAlreadyAddedToDT[&MI] = Reg;
          // This MI is unused and will be removed. If the MI uses
          // const_composite, it will be unused and should be removed too.
          assert(MI.getOperand(2).isReg() && "Reg operand is expected");
          MachineInstr *SrcMI = MRI.getVRegDef(MI.getOperand(2).getReg());
          if (SrcMI && isSpvIntrinsic(*SrcMI, Intrinsic::spv_const_composite))
            ToEraseComposites.push_back(SrcMI);
        }
      }
    }
  }
  for (MachineInstr *MI : ToErase) {
    Register Reg = MI->getOperand(2).getReg();
    auto It = RegsAlreadyAddedToDT.find(MI);
    if (It != RegsAlreadyAddedToDT.end())
      Reg = It->second;
    auto *RC = MRI.getRegClassOrNull(MI->getOperand(0).getReg());
    if (!MRI.getRegClassOrNull(Reg) && RC)
      MRI.setRegClass(Reg, RC);
    MRI.replaceRegWith(MI->getOperand(0).getReg(), Reg);
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
  for (MachineInstr *MI : ToEraseComposites) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

static void foldConstantsIntoIntrinsics(MachineFunction &MF,
                                        SPIRVGlobalRegistry *GR,
                                        MachineIRBuilder MIB) {
  SmallVector<MachineInstr *, 64> ToErase;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_assign_name))
        continue;
      const MDNode *MD = MI.getOperand(2).getMetadata();
      StringRef ValueName = cast<MDString>(MD->getOperand(0))->getString();
      if (ValueName.size() > 0) {
        MIB.setInsertPt(*MI.getParent(), MI);
        buildOpName(MI.getOperand(1).getReg(), ValueName, MIB);
      }
      ToErase.push_back(&MI);
    }
    for (MachineInstr *MI : ToErase) {
      GR->invalidateMachineInstr(MI);
      MI->eraseFromParent();
    }
    ToErase.clear();
  }
}

static MachineInstr *findAssignTypeInstr(Register Reg,
                                         MachineRegisterInfo *MRI) {
  for (MachineRegisterInfo::use_instr_iterator I = MRI->use_instr_begin(Reg),
                                               IE = MRI->use_instr_end();
       I != IE; ++I) {
    MachineInstr *UseMI = &*I;
    if ((isSpvIntrinsic(*UseMI, Intrinsic::spv_assign_ptr_type) ||
         isSpvIntrinsic(*UseMI, Intrinsic::spv_assign_type)) &&
        UseMI->getOperand(1).getReg() == Reg)
      return UseMI;
  }
  return nullptr;
}

static void buildOpBitcast(SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                           Register ResVReg, Register OpReg) {
  SPIRVType *ResType = GR->getSPIRVTypeForVReg(ResVReg);
  SPIRVType *OpType = GR->getSPIRVTypeForVReg(OpReg);
  assert(ResType && OpType && "Operand types are expected");
  if (!GR->isBitcastCompatible(ResType, OpType))
    report_fatal_error("incompatible result and operand types in a bitcast");
  MachineRegisterInfo *MRI = MIB.getMRI();
  if (!MRI->getRegClassOrNull(ResVReg))
    MRI->setRegClass(ResVReg, GR->getRegClass(ResType));
  if (ResType == OpType)
    MIB.buildInstr(TargetOpcode::COPY).addDef(ResVReg).addUse(OpReg);
  else
    MIB.buildInstr(SPIRV::OpBitcast)
        .addDef(ResVReg)
        .addUse(GR->getSPIRVTypeID(ResType))
        .addUse(OpReg);
}

// We do instruction selections early instead of calling MIB.buildBitcast()
// generating the general op code G_BITCAST. When MachineVerifier validates
// G_BITCAST we see a check of a kind: if Source Type is equal to Destination
// Type then report error "bitcast must change the type". This doesn't take into
// account the notion of a typed pointer that is important for SPIR-V where a
// user may and should use bitcast between pointers with different pointee types
// (https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitcast).
// It's important for correct lowering in SPIR-V, because interpretation of the
// data type is not left to instructions that utilize the pointer, but encoded
// by the pointer declaration, and the SPIRV target can and must handle the
// declaration and use of pointers that specify the type of data they point to.
// It's not feasible to improve validation of G_BITCAST using just information
// provided by low level types of source and destination. Therefore we don't
// produce G_BITCAST as the general op code with semantics different from
// OpBitcast, but rather lower to OpBitcast immediately. As for now, the only
// difference would be that CombinerHelper couldn't transform known patterns
// around G_BUILD_VECTOR. See discussion
// in https://github.com/llvm/llvm-project/pull/110270 for even more context.
static void selectOpBitcasts(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                             MachineIRBuilder MIB) {
  SmallVector<MachineInstr *, 16> ToErase;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != TargetOpcode::G_BITCAST)
        continue;
      MIB.setInsertPt(*MI.getParent(), MI);
      buildOpBitcast(GR, MIB, MI.getOperand(0).getReg(),
                     MI.getOperand(1).getReg());
      ToErase.push_back(&MI);
    }
  }
  for (MachineInstr *MI : ToErase) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

static void insertBitcasts(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                           MachineIRBuilder MIB) {
  // Get access to information about available extensions
  const SPIRVSubtarget *ST =
      static_cast<const SPIRVSubtarget *>(&MIB.getMF().getSubtarget());
  SmallVector<MachineInstr *, 10> ToErase;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_bitcast) &&
          !isSpvIntrinsic(MI, Intrinsic::spv_ptrcast))
        continue;
      assert(MI.getOperand(2).isReg());
      MIB.setInsertPt(*MI.getParent(), MI);
      ToErase.push_back(&MI);
      if (isSpvIntrinsic(MI, Intrinsic::spv_bitcast)) {
        MIB.buildBitcast(MI.getOperand(0).getReg(), MI.getOperand(2).getReg());
        continue;
      }
      Register Def = MI.getOperand(0).getReg();
      Register Source = MI.getOperand(2).getReg();
      Type *ElemTy = getMDOperandAsType(MI.getOperand(3).getMetadata(), 0);
      SPIRVType *AssignedPtrType = GR->getOrCreateSPIRVPointerType(
          ElemTy, MI,
          addressSpaceToStorageClass(MI.getOperand(4).getImm(), *ST));

      // If the ptrcast would be redundant, replace all uses with the source
      // register.
      MachineRegisterInfo *MRI = MIB.getMRI();
      if (GR->getSPIRVTypeForVReg(Source) == AssignedPtrType) {
        // Erase Def's assign type instruction if we are going to replace Def.
        if (MachineInstr *AssignMI = findAssignTypeInstr(Def, MRI))
          ToErase.push_back(AssignMI);
        MRI->replaceRegWith(Def, Source);
      } else {
        if (!GR->getSPIRVTypeForVReg(Def, &MF))
          GR->assignSPIRVTypeToVReg(AssignedPtrType, Def, MF);
        MIB.buildBitcast(Def, Source);
      }
    }
  }
  for (MachineInstr *MI : ToErase) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

// Translating GV, IRTranslator sometimes generates following IR:
//   %1 = G_GLOBAL_VALUE
//   %2 = COPY %1
//   %3 = G_ADDRSPACE_CAST %2
//
// or
//
//  %1 = G_ZEXT %2
//  G_MEMCPY ... %2 ...
//
// New registers have no SPIRVType and no register class info.
//
// Set SPIRVType for GV, propagate it from GV to other instructions,
// also set register classes.
static SPIRVType *propagateSPIRVType(MachineInstr *MI, SPIRVGlobalRegistry *GR,
                                     MachineRegisterInfo &MRI,
                                     MachineIRBuilder &MIB) {
  SPIRVType *SpvType = nullptr;
  assert(MI && "Machine instr is expected");
  if (MI->getOperand(0).isReg()) {
    Register Reg = MI->getOperand(0).getReg();
    SpvType = GR->getSPIRVTypeForVReg(Reg);
    if (!SpvType) {
      switch (MI->getOpcode()) {
      case TargetOpcode::G_FCONSTANT:
      case TargetOpcode::G_CONSTANT: {
        MIB.setInsertPt(*MI->getParent(), MI);
        Type *Ty = MI->getOperand(1).getCImm()->getType();
        SpvType = GR->getOrCreateSPIRVType(
            Ty, MIB, SPIRV::AccessQualifier::ReadWrite, true);
        break;
      }
      case TargetOpcode::G_GLOBAL_VALUE: {
        MIB.setInsertPt(*MI->getParent(), MI);
        const GlobalValue *Global = MI->getOperand(1).getGlobal();
        Type *ElementTy = toTypedPointer(GR->getDeducedGlobalValueType(Global));
        auto *Ty = TypedPointerType::get(ElementTy,
                                         Global->getType()->getAddressSpace());
        SpvType = GR->getOrCreateSPIRVType(
            Ty, MIB, SPIRV::AccessQualifier::ReadWrite, true);
        break;
      }
      case TargetOpcode::G_ANYEXT:
      case TargetOpcode::G_SEXT:
      case TargetOpcode::G_ZEXT: {
        if (MI->getOperand(1).isReg()) {
          if (MachineInstr *DefInstr =
                  MRI.getVRegDef(MI->getOperand(1).getReg())) {
            if (SPIRVType *Def = propagateSPIRVType(DefInstr, GR, MRI, MIB)) {
              unsigned CurrentBW = GR->getScalarOrVectorBitWidth(Def);
              unsigned ExpectedBW =
                  std::max(MRI.getType(Reg).getScalarSizeInBits(), CurrentBW);
              unsigned NumElements = GR->getScalarOrVectorComponentCount(Def);
              SpvType = GR->getOrCreateSPIRVIntegerType(ExpectedBW, MIB);
              if (NumElements > 1)
                SpvType = GR->getOrCreateSPIRVVectorType(SpvType, NumElements,
                                                         MIB, true);
            }
          }
        }
        break;
      }
      case TargetOpcode::G_PTRTOINT:
        SpvType = GR->getOrCreateSPIRVIntegerType(
            MRI.getType(Reg).getScalarSizeInBits(), MIB);
        break;
      case TargetOpcode::G_TRUNC:
      case TargetOpcode::G_ADDRSPACE_CAST:
      case TargetOpcode::G_PTR_ADD:
      case TargetOpcode::COPY: {
        MachineOperand &Op = MI->getOperand(1);
        MachineInstr *Def = Op.isReg() ? MRI.getVRegDef(Op.getReg()) : nullptr;
        if (Def)
          SpvType = propagateSPIRVType(Def, GR, MRI, MIB);
        break;
      }
      default:
        break;
      }
      if (SpvType) {
        // check if the address space needs correction
        LLT RegType = MRI.getType(Reg);
        if (SpvType->getOpcode() == SPIRV::OpTypePointer &&
            RegType.isPointer() &&
            storageClassToAddressSpace(GR->getPointerStorageClass(SpvType)) !=
                RegType.getAddressSpace()) {
          const SPIRVSubtarget &ST =
              MI->getParent()->getParent()->getSubtarget<SPIRVSubtarget>();
          auto TSC = addressSpaceToStorageClass(RegType.getAddressSpace(), ST);
          SpvType = GR->changePointerStorageClass(SpvType, TSC, *MI);
        }
        GR->assignSPIRVTypeToVReg(SpvType, Reg, MIB.getMF());
      }
      if (!MRI.getRegClassOrNull(Reg))
        MRI.setRegClass(Reg, SpvType ? GR->getRegClass(SpvType)
                                     : &SPIRV::iIDRegClass);
    }
  }
  return SpvType;
}

// To support current approach and limitations wrt. bit width here we widen a
// scalar register with a bit width greater than 1 to valid sizes and cap it to
// 64 width.
static unsigned widenBitWidthToNextPow2(unsigned BitWidth) {
  if (BitWidth == 1)
    return 1; // No need to widen 1-bit values
  return std::min(std::max(1u << Log2_32_Ceil(BitWidth), 8u), 64u);
}

static void widenScalarType(Register Reg, MachineRegisterInfo &MRI) {
  LLT RegType = MRI.getType(Reg);
  if (!RegType.isScalar())
    return;
  unsigned CurrentWidth = RegType.getScalarSizeInBits();
  unsigned NewWidth = widenBitWidthToNextPow2(CurrentWidth);
  if (NewWidth != CurrentWidth)
    MRI.setType(Reg, LLT::scalar(NewWidth));
}

static void widenCImmType(MachineOperand &MOP) {
  const ConstantInt *CImmVal = MOP.getCImm();
  unsigned CurrentWidth = CImmVal->getBitWidth();
  unsigned NewWidth = widenBitWidthToNextPow2(CurrentWidth);
  if (NewWidth != CurrentWidth) {
    // Replace the immediate value with the widened version
    MOP.setCImm(ConstantInt::get(CImmVal->getType()->getContext(),
                                 CImmVal->getValue().zextOrTrunc(NewWidth)));
  }
}

static void setInsertPtAfterDef(MachineIRBuilder &MIB, MachineInstr *Def) {
  MachineBasicBlock &MBB = *Def->getParent();
  MachineBasicBlock::iterator DefIt =
      Def->getNextNode() ? Def->getNextNode()->getIterator() : MBB.end();
  // Skip all the PHI and debug instructions.
  while (DefIt != MBB.end() &&
         (DefIt->isPHI() || DefIt->isDebugOrPseudoInstr()))
    DefIt = std::next(DefIt);
  MIB.setInsertPt(MBB, DefIt);
}

namespace llvm {
void insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpvType,
                       SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                       MachineRegisterInfo &MRI) {
  assert((Ty || SpvType) && "Either LLVM or SPIRV type is expected.");
  MachineInstr *Def = MRI.getVRegDef(Reg);
  setInsertPtAfterDef(MIB, Def);
  if (!SpvType)
    SpvType = GR->getOrCreateSPIRVType(Ty, MIB,
                                       SPIRV::AccessQualifier::ReadWrite, true);

  if (!isTypeFoldingSupported(Def->getOpcode())) {
    // No need to generate SPIRV::ASSIGN_TYPE pseudo-instruction
    if (!MRI.getRegClassOrNull(Reg))
      MRI.setRegClass(Reg, GR->getRegClass(SpvType));
    if (!MRI.getType(Reg).isValid())
      MRI.setType(Reg, GR->getRegType(SpvType));
    GR->assignSPIRVTypeToVReg(SpvType, Reg, MIB.getMF());
    return;
  }

  // Tablegen definition assumes SPIRV::ASSIGN_TYPE pseudo-instruction is
  // present after each auto-folded instruction to take a type reference from.
  Register NewReg = MRI.createGenericVirtualRegister(MRI.getType(Reg));
  if (auto *RC = MRI.getRegClassOrNull(Reg)) {
    MRI.setRegClass(NewReg, RC);
  } else {
    auto RegClass = GR->getRegClass(SpvType);
    MRI.setRegClass(NewReg, RegClass);
    MRI.setRegClass(Reg, RegClass);
  }
  GR->assignSPIRVTypeToVReg(SpvType, Reg, MIB.getMF());
  // This is to make it convenient for Legalizer to get the SPIRVType
  // when processing the actual MI (i.e. not pseudo one).
  GR->assignSPIRVTypeToVReg(SpvType, NewReg, MIB.getMF());
  // Copy MIFlags from Def to ASSIGN_TYPE instruction. It's required to keep
  // the flags after instruction selection.
  const uint32_t Flags = Def->getFlags();
  MIB.buildInstr(SPIRV::ASSIGN_TYPE)
      .addDef(Reg)
      .addUse(NewReg)
      .addUse(GR->getSPIRVTypeID(SpvType))
      .setMIFlags(Flags);
  for (unsigned I = 0, E = Def->getNumDefs(); I != E; ++I) {
    MachineOperand &MO = Def->getOperand(I);
    if (MO.getReg() == Reg) {
      MO.setReg(NewReg);
      break;
    }
  }
}

void processInstr(MachineInstr &MI, MachineIRBuilder &MIB,
                  MachineRegisterInfo &MRI, SPIRVGlobalRegistry *GR,
                  SPIRVType *KnownResType) {
  MIB.setInsertPt(*MI.getParent(), MI.getIterator());
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || Op.isDef())
      continue;
    Register OpReg = Op.getReg();
    SPIRVType *SpvType = GR->getSPIRVTypeForVReg(OpReg);
    if (!SpvType && KnownResType) {
      SpvType = KnownResType;
      GR->assignSPIRVTypeToVReg(KnownResType, OpReg, *MI.getMF());
    }
    assert(SpvType);
    if (!MRI.getRegClassOrNull(OpReg))
      MRI.setRegClass(OpReg, GR->getRegClass(SpvType));
    if (!MRI.getType(OpReg).isValid())
      MRI.setType(OpReg, GR->getRegType(SpvType));
  }
}
} // namespace llvm

static void
generateAssignInstrs(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                     MachineIRBuilder MIB,
                     DenseMap<MachineInstr *, Type *> &TargetExtConstTypes) {
  // Get access to information about available extensions
  const SPIRVSubtarget *ST =
      static_cast<const SPIRVSubtarget *>(&MIB.getMF().getSubtarget());

  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 10> ToErase;
  DenseMap<MachineInstr *, Register> RegsAlreadyAddedToDT;

  bool IsExtendedInts =
      ST->canUseExtension(
          SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers) ||
      ST->canUseExtension(SPIRV::Extension::SPV_KHR_bit_instructions) ||
      ST->canUseExtension(SPIRV::Extension::SPV_INTEL_int4);

  for (MachineBasicBlock *MBB : post_order(&MF)) {
    if (MBB->empty())
      continue;

    bool ReachedBegin = false;
    for (auto MII = std::prev(MBB->end()), Begin = MBB->begin();
         !ReachedBegin;) {
      MachineInstr &MI = *MII;
      unsigned MIOp = MI.getOpcode();

      if (!IsExtendedInts) {
        // validate bit width of scalar registers and constant immediates
        for (auto &MOP : MI.operands()) {
          if (MOP.isReg())
            widenScalarType(MOP.getReg(), MRI);
          else if (MOP.isCImm())
            widenCImmType(MOP);
        }
      }

      if (isSpvIntrinsic(MI, Intrinsic::spv_assign_ptr_type)) {
        Register Reg = MI.getOperand(1).getReg();
        MIB.setInsertPt(*MI.getParent(), MI.getIterator());
        Type *ElementTy = getMDOperandAsType(MI.getOperand(2).getMetadata(), 0);
        SPIRVType *AssignedPtrType = GR->getOrCreateSPIRVPointerType(
            ElementTy, MI,
            addressSpaceToStorageClass(MI.getOperand(3).getImm(), *ST));
        MachineInstr *Def = MRI.getVRegDef(Reg);
        assert(Def && "Expecting an instruction that defines the register");
        // G_GLOBAL_VALUE already has type info.
        if (Def->getOpcode() != TargetOpcode::G_GLOBAL_VALUE &&
            Def->getOpcode() != SPIRV::ASSIGN_TYPE)
          insertAssignInstr(Reg, nullptr, AssignedPtrType, GR, MIB,
                            MF.getRegInfo());
        ToErase.push_back(&MI);
      } else if (isSpvIntrinsic(MI, Intrinsic::spv_assign_type)) {
        Register Reg = MI.getOperand(1).getReg();
        Type *Ty = getMDOperandAsType(MI.getOperand(2).getMetadata(), 0);
        MachineInstr *Def = MRI.getVRegDef(Reg);
        assert(Def && "Expecting an instruction that defines the register");
        // G_GLOBAL_VALUE already has type info.
        if (Def->getOpcode() != TargetOpcode::G_GLOBAL_VALUE &&
            Def->getOpcode() != SPIRV::ASSIGN_TYPE)
          insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MF.getRegInfo());
        ToErase.push_back(&MI);
      } else if (MIOp == TargetOpcode::FAKE_USE && MI.getNumOperands() > 0) {
        MachineInstr *MdMI = MI.getPrevNode();
        if (MdMI && isSpvIntrinsic(*MdMI, Intrinsic::spv_value_md)) {
          // It's an internal service info from before IRTranslator passes.
          MachineInstr *Def = getVRegDef(MRI, MI.getOperand(0).getReg());
          for (unsigned I = 1, E = MI.getNumOperands(); I != E && Def; ++I)
            if (getVRegDef(MRI, MI.getOperand(I).getReg()) != Def)
              Def = nullptr;
          if (Def) {
            const MDNode *MD = MdMI->getOperand(1).getMetadata();
            StringRef ValueName =
                cast<MDString>(MD->getOperand(1))->getString();
            const MDNode *TypeMD = cast<MDNode>(MD->getOperand(0));
            Type *ValueTy = getMDOperandAsType(TypeMD, 0);
            GR->addValueAttrs(Def, std::make_pair(ValueTy, ValueName.str()));
          }
          ToErase.push_back(MdMI);
        }
        ToErase.push_back(&MI);
      } else if (MIOp == TargetOpcode::G_CONSTANT ||
                 MIOp == TargetOpcode::G_FCONSTANT ||
                 MIOp == TargetOpcode::G_BUILD_VECTOR) {
        // %rc = G_CONSTANT ty Val
        // ===>
        // %cty = OpType* ty
        // %rctmp = G_CONSTANT ty Val
        // %rc = ASSIGN_TYPE %rctmp, %cty
        Register Reg = MI.getOperand(0).getReg();
        bool NeedAssignType = true;
        if (MRI.hasOneUse(Reg)) {
          MachineInstr &UseMI = *MRI.use_instr_begin(Reg);
          if (isSpvIntrinsic(UseMI, Intrinsic::spv_assign_type) ||
              isSpvIntrinsic(UseMI, Intrinsic::spv_assign_name))
            continue;
          if (UseMI.getOpcode() == SPIRV::ASSIGN_TYPE)
            NeedAssignType = false;
        }
        Type *Ty = nullptr;
        if (MIOp == TargetOpcode::G_CONSTANT) {
          auto TargetExtIt = TargetExtConstTypes.find(&MI);
          Ty = TargetExtIt == TargetExtConstTypes.end()
                   ? MI.getOperand(1).getCImm()->getType()
                   : TargetExtIt->second;
          const ConstantInt *OpCI = MI.getOperand(1).getCImm();
          // TODO: we may wish to analyze here if OpCI is zero and LLT RegType =
          // MRI.getType(Reg); RegType.isPointer() is true, so that we observe
          // at this point not i64/i32 constant but null pointer in the
          // corresponding address space of RegType.getAddressSpace(). This may
          // help to successfully validate the case when a OpConstantComposite's
          // constituent has type that does not match Result Type of
          // OpConstantComposite (see, for example,
          // pointers/PtrCast-null-in-OpSpecConstantOp.ll).
          Register PrimaryReg = GR->find(OpCI, &MF);
          if (!PrimaryReg.isValid()) {
            GR->add(OpCI, &MI);
          } else if (PrimaryReg != Reg &&
                     MRI.getType(Reg) == MRI.getType(PrimaryReg)) {
            auto *RCReg = MRI.getRegClassOrNull(Reg);
            auto *RCPrimary = MRI.getRegClassOrNull(PrimaryReg);
            if (!RCReg || RCPrimary == RCReg) {
              RegsAlreadyAddedToDT[&MI] = PrimaryReg;
              ToErase.push_back(&MI);
              NeedAssignType = false;
            }
          }
        } else if (MIOp == TargetOpcode::G_FCONSTANT) {
          Ty = MI.getOperand(1).getFPImm()->getType();
        } else {
          assert(MIOp == TargetOpcode::G_BUILD_VECTOR);
          Type *ElemTy = nullptr;
          MachineInstr *ElemMI = MRI.getVRegDef(MI.getOperand(1).getReg());
          assert(ElemMI);

          if (ElemMI->getOpcode() == TargetOpcode::G_CONSTANT) {
            ElemTy = ElemMI->getOperand(1).getCImm()->getType();
          } else if (ElemMI->getOpcode() == TargetOpcode::G_FCONSTANT) {
            ElemTy = ElemMI->getOperand(1).getFPImm()->getType();
          } else {
            if (const SPIRVType *ElemSpvType =
                    GR->getSPIRVTypeForVReg(MI.getOperand(1).getReg(), &MF))
              ElemTy = const_cast<Type *>(GR->getTypeForSPIRVType(ElemSpvType));
            if (!ElemTy) {
              // There may be a case when we already know Reg's type.
              MachineInstr *NextMI = MI.getNextNode();
              if (!NextMI || NextMI->getOpcode() != SPIRV::ASSIGN_TYPE ||
                  NextMI->getOperand(1).getReg() != Reg)
                llvm_unreachable("Unexpected opcode");
            }
          }
          if (ElemTy)
            Ty = VectorType::get(
                ElemTy, MI.getNumExplicitOperands() - MI.getNumExplicitDefs(),
                false);
          else
            NeedAssignType = false;
        }
        if (NeedAssignType)
          insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MRI);
      } else if (MIOp == TargetOpcode::G_GLOBAL_VALUE) {
        propagateSPIRVType(&MI, GR, MRI, MIB);
      }

      if (MII == Begin)
        ReachedBegin = true;
      else
        --MII;
    }
  }
  for (MachineInstr *MI : ToErase) {
    auto It = RegsAlreadyAddedToDT.find(MI);
    if (It != RegsAlreadyAddedToDT.end())
      MRI.replaceRegWith(MI->getOperand(0).getReg(), It->second);
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }

  // Address the case when IRTranslator introduces instructions with new
  // registers without SPIRVType associated.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case TargetOpcode::G_TRUNC:
      case TargetOpcode::G_ANYEXT:
      case TargetOpcode::G_SEXT:
      case TargetOpcode::G_ZEXT:
      case TargetOpcode::G_PTRTOINT:
      case TargetOpcode::COPY:
      case TargetOpcode::G_ADDRSPACE_CAST:
        propagateSPIRVType(&MI, GR, MRI, MIB);
        break;
      }
    }
  }
}

static void processInstrsWithTypeFolding(MachineFunction &MF,
                                         SPIRVGlobalRegistry *GR,
                                         MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (isTypeFoldingSupported(MI.getOpcode()))
        processInstr(MI, MIB, MRI, GR, nullptr);
}

static Register
collectInlineAsmInstrOperands(MachineInstr *MI,
                              SmallVector<unsigned, 4> *Ops = nullptr) {
  Register DefReg;
  unsigned StartOp = InlineAsm::MIOp_FirstOperand,
           AsmDescOp = InlineAsm::MIOp_FirstOperand;
  for (unsigned Idx = StartOp, MISz = MI->getNumOperands(); Idx != MISz;
       ++Idx) {
    const MachineOperand &MO = MI->getOperand(Idx);
    if (MO.isMetadata())
      continue;
    if (Idx == AsmDescOp && MO.isImm()) {
      // compute the index of the next operand descriptor
      const InlineAsm::Flag F(MO.getImm());
      AsmDescOp += 1 + F.getNumOperandRegisters();
      continue;
    }
    if (MO.isReg() && MO.isDef()) {
      if (!Ops)
        return MO.getReg();
      else
        DefReg = MO.getReg();
    } else if (Ops) {
      Ops->push_back(Idx);
    }
  }
  return DefReg;
}

static void
insertInlineAsmProcess(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                       const SPIRVSubtarget &ST, MachineIRBuilder MIRBuilder,
                       const SmallVector<MachineInstr *> &ToProcess) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register AsmTargetReg;
  for (unsigned i = 0, Sz = ToProcess.size(); i + 1 < Sz; i += 2) {
    MachineInstr *I1 = ToProcess[i], *I2 = ToProcess[i + 1];
    assert(isSpvIntrinsic(*I1, Intrinsic::spv_inline_asm) && I2->isInlineAsm());
    MIRBuilder.setInsertPt(*I2->getParent(), *I2);

    if (!AsmTargetReg.isValid()) {
      // define vendor specific assembly target or dialect
      AsmTargetReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
      MRI.setRegClass(AsmTargetReg, &SPIRV::iIDRegClass);
      auto AsmTargetMIB =
          MIRBuilder.buildInstr(SPIRV::OpAsmTargetINTEL).addDef(AsmTargetReg);
      addStringImm(ST.getTargetTripleAsStr(), AsmTargetMIB);
      GR->add(AsmTargetMIB.getInstr(), AsmTargetMIB);
    }

    // create types
    const MDNode *IAMD = I1->getOperand(1).getMetadata();
    FunctionType *FTy = cast<FunctionType>(getMDOperandAsType(IAMD, 0));
    SmallVector<SPIRVType *, 4> ArgTypes;
    for (const auto &ArgTy : FTy->params())
      ArgTypes.push_back(GR->getOrCreateSPIRVType(
          ArgTy, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, true));
    SPIRVType *RetType =
        GR->getOrCreateSPIRVType(FTy->getReturnType(), MIRBuilder,
                                 SPIRV::AccessQualifier::ReadWrite, true);
    SPIRVType *FuncType = GR->getOrCreateOpTypeFunctionWithArgs(
        FTy, RetType, ArgTypes, MIRBuilder);

    // define vendor specific assembly instructions string
    Register AsmReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
    MRI.setRegClass(AsmReg, &SPIRV::iIDRegClass);
    auto AsmMIB = MIRBuilder.buildInstr(SPIRV::OpAsmINTEL)
                      .addDef(AsmReg)
                      .addUse(GR->getSPIRVTypeID(RetType))
                      .addUse(GR->getSPIRVTypeID(FuncType))
                      .addUse(AsmTargetReg);
    // inline asm string:
    addStringImm(I2->getOperand(InlineAsm::MIOp_AsmString).getSymbolName(),
                 AsmMIB);
    // inline asm constraint string:
    addStringImm(cast<MDString>(I1->getOperand(2).getMetadata()->getOperand(0))
                     ->getString(),
                 AsmMIB);
    GR->add(AsmMIB.getInstr(), AsmMIB);

    // calls the inline assembly instruction
    unsigned ExtraInfo = I2->getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
    if (ExtraInfo & InlineAsm::Extra_HasSideEffects)
      MIRBuilder.buildInstr(SPIRV::OpDecorate)
          .addUse(AsmReg)
          .addImm(static_cast<uint32_t>(SPIRV::Decoration::SideEffectsINTEL));

    Register DefReg = collectInlineAsmInstrOperands(I2);
    if (!DefReg.isValid()) {
      DefReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
      MRI.setRegClass(DefReg, &SPIRV::iIDRegClass);
      SPIRVType *VoidType = GR->getOrCreateSPIRVType(
          Type::getVoidTy(MF.getFunction().getContext()), MIRBuilder,
          SPIRV::AccessQualifier::ReadWrite, true);
      GR->assignSPIRVTypeToVReg(VoidType, DefReg, MF);
    }

    auto AsmCall = MIRBuilder.buildInstr(SPIRV::OpAsmCallINTEL)
                       .addDef(DefReg)
                       .addUse(GR->getSPIRVTypeID(RetType))
                       .addUse(AsmReg);
    for (unsigned IntrIdx = 3; IntrIdx < I1->getNumOperands(); ++IntrIdx)
      AsmCall.addUse(I1->getOperand(IntrIdx).getReg());
  }
  for (MachineInstr *MI : ToProcess) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

static void insertInlineAsm(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                            const SPIRVSubtarget &ST,
                            MachineIRBuilder MIRBuilder) {
  SmallVector<MachineInstr *> ToProcess;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isSpvIntrinsic(MI, Intrinsic::spv_inline_asm) ||
          MI.getOpcode() == TargetOpcode::INLINEASM)
        ToProcess.push_back(&MI);
    }
  }
  if (ToProcess.size() == 0)
    return;

  if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_inline_assembly))
    report_fatal_error("Inline assembly instructions require the "
                       "following SPIR-V extension: SPV_INTEL_inline_assembly",
                       false);

  insertInlineAsmProcess(MF, GR, ST, MIRBuilder, ToProcess);
}

static uint32_t convertFloatToSPIRVWord(float F) {
  union {
    float F;
    uint32_t Spir;
  } FPMaxError;
  FPMaxError.F = F;
  return FPMaxError.Spir;
}

static void insertSpirvDecorations(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                                   MachineIRBuilder MIB) {
  SmallVector<MachineInstr *, 10> ToErase;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_assign_decoration) &&
          !isSpvIntrinsic(MI, Intrinsic::spv_assign_aliasing_decoration) &&
          !isSpvIntrinsic(MI, Intrinsic::spv_assign_fpmaxerror_decoration))
        continue;
      MIB.setInsertPt(*MI.getParent(), MI.getNextNode());
      if (isSpvIntrinsic(MI, Intrinsic::spv_assign_decoration)) {
        buildOpSpirvDecorations(MI.getOperand(1).getReg(), MIB,
                                MI.getOperand(2).getMetadata());
      } else if (isSpvIntrinsic(MI,
                                Intrinsic::spv_assign_fpmaxerror_decoration)) {
        ConstantFP *OpV = mdconst::dyn_extract<ConstantFP>(
            MI.getOperand(2).getMetadata()->getOperand(0));
        uint32_t OpValue =
            convertFloatToSPIRVWord(OpV->getValueAPF().convertToFloat());

        buildOpDecorate(MI.getOperand(1).getReg(), MIB,
                        SPIRV::Decoration::FPMaxErrorDecorationINTEL,
                        {OpValue});
      } else {
        GR->buildMemAliasingOpDecorate(MI.getOperand(1).getReg(), MIB,
                                       MI.getOperand(2).getImm(),
                                       MI.getOperand(3).getMetadata());
      }

      ToErase.push_back(&MI);
    }
  }
  for (MachineInstr *MI : ToErase) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

// LLVM allows the switches to use registers as cases, while SPIR-V required
// those to be immediate values. This function replaces such operands with the
// equivalent immediate constant.
static void processSwitchesConstants(MachineFunction &MF,
                                     SPIRVGlobalRegistry *GR,
                                     MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_switch))
        continue;

      SmallVector<MachineOperand, 8> NewOperands;
      NewOperands.push_back(MI.getOperand(0)); // Opcode
      NewOperands.push_back(MI.getOperand(1)); // Condition
      NewOperands.push_back(MI.getOperand(2)); // Default
      for (unsigned i = 3; i < MI.getNumOperands(); i += 2) {
        Register Reg = MI.getOperand(i).getReg();
        MachineInstr *ConstInstr = getDefInstrMaybeConstant(Reg, &MRI);
        NewOperands.push_back(
            MachineOperand::CreateCImm(ConstInstr->getOperand(1).getCImm()));

        NewOperands.push_back(MI.getOperand(i + 1));
      }

      assert(MI.getNumOperands() == NewOperands.size());
      while (MI.getNumOperands() > 0)
        MI.removeOperand(0);
      for (auto &MO : NewOperands)
        MI.addOperand(MO);
    }
  }
}

// Some instructions are used during CodeGen but should never be emitted.
// Cleaning up those.
static void cleanupHelperInstructions(MachineFunction &MF,
                                      SPIRVGlobalRegistry *GR) {
  SmallVector<MachineInstr *, 8> ToEraseMI;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isSpvIntrinsic(MI, Intrinsic::spv_track_constant) ||
          MI.getOpcode() == TargetOpcode::G_BRINDIRECT)
        ToEraseMI.push_back(&MI);
    }
  }

  for (MachineInstr *MI : ToEraseMI) {
    GR->invalidateMachineInstr(MI);
    MI->eraseFromParent();
  }
}

// Find all usages of G_BLOCK_ADDR in our intrinsics and replace those
// operands/registers by the actual MBB it references.
static void processBlockAddr(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                             MachineIRBuilder MIB) {
  // Gather the reverse-mapping BB -> MBB.
  DenseMap<const BasicBlock *, MachineBasicBlock *> BB2MBB;
  for (MachineBasicBlock &MBB : MF)
    BB2MBB[MBB.getBasicBlock()] = &MBB;

  // Gather instructions requiring patching. For now, only those can use
  // G_BLOCK_ADDR.
  SmallVector<MachineInstr *, 8> InstructionsToPatch;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isSpvIntrinsic(MI, Intrinsic::spv_switch) ||
          isSpvIntrinsic(MI, Intrinsic::spv_loop_merge) ||
          isSpvIntrinsic(MI, Intrinsic::spv_selection_merge))
        InstructionsToPatch.push_back(&MI);
    }
  }

  // For each instruction to fix, we replace all the G_BLOCK_ADDR operands by
  // the actual MBB it references. Once those references have been updated, we
  // can cleanup remaining G_BLOCK_ADDR references.
  SmallPtrSet<MachineBasicBlock *, 8> ClearAddressTaken;
  SmallPtrSet<MachineInstr *, 8> ToEraseMI;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineInstr *MI : InstructionsToPatch) {
    SmallVector<MachineOperand, 8> NewOps;
    for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
      // The operand is not a register, keep as-is.
      if (!MI->getOperand(i).isReg()) {
        NewOps.push_back(MI->getOperand(i));
        continue;
      }

      Register Reg = MI->getOperand(i).getReg();
      MachineInstr *BuildMBB = MRI.getVRegDef(Reg);
      // The register is not the result of G_BLOCK_ADDR, keep as-is.
      if (!BuildMBB || BuildMBB->getOpcode() != TargetOpcode::G_BLOCK_ADDR) {
        NewOps.push_back(MI->getOperand(i));
        continue;
      }

      assert(BuildMBB && BuildMBB->getOpcode() == TargetOpcode::G_BLOCK_ADDR &&
             BuildMBB->getOperand(1).isBlockAddress() &&
             BuildMBB->getOperand(1).getBlockAddress());
      BasicBlock *BB =
          BuildMBB->getOperand(1).getBlockAddress()->getBasicBlock();
      auto It = BB2MBB.find(BB);
      if (It == BB2MBB.end())
        report_fatal_error("cannot find a machine basic block by a basic block "
                           "in a switch statement");
      MachineBasicBlock *ReferencedBlock = It->second;
      NewOps.push_back(MachineOperand::CreateMBB(ReferencedBlock));

      ClearAddressTaken.insert(ReferencedBlock);
      ToEraseMI.insert(BuildMBB);
    }

    // Replace the operands.
    assert(MI->getNumOperands() == NewOps.size());
    while (MI->getNumOperands() > 0)
      MI->removeOperand(0);
    for (auto &MO : NewOps)
      MI->addOperand(MO);

    if (MachineInstr *Next = MI->getNextNode()) {
      if (isSpvIntrinsic(*Next, Intrinsic::spv_track_constant)) {
        ToEraseMI.insert(Next);
        Next = MI->getNextNode();
      }
      if (Next && Next->getOpcode() == TargetOpcode::G_BRINDIRECT)
        ToEraseMI.insert(Next);
    }
  }

  // BlockAddress operands were used to keep information between passes,
  // let's undo the "address taken" status to reflect that Succ doesn't
  // actually correspond to an IR-level basic block.
  for (MachineBasicBlock *Succ : ClearAddressTaken)
    Succ->setAddressTakenIRBlock(nullptr);

  // If we just delete G_BLOCK_ADDR instructions with BlockAddress operands,
  // this leaves their BasicBlock counterparts in a "address taken" status. This
  // would make AsmPrinter to generate a series of unneeded labels of a "Address
  // of block that was removed by CodeGen" kind. Let's first ensure that we
  // don't have a dangling BlockAddress constants by zapping the BlockAddress
  // nodes, and only after that proceed with erasing G_BLOCK_ADDR instructions.
  Constant *Replacement =
      ConstantInt::get(Type::getInt32Ty(MF.getFunction().getContext()), 1);
  for (MachineInstr *BlockAddrI : ToEraseMI) {
    if (BlockAddrI->getOpcode() == TargetOpcode::G_BLOCK_ADDR) {
      BlockAddress *BA = const_cast<BlockAddress *>(
          BlockAddrI->getOperand(1).getBlockAddress());
      BA->replaceAllUsesWith(
          ConstantExpr::getIntToPtr(Replacement, BA->getType()));
      BA->destroyConstant();
    }
    GR->invalidateMachineInstr(BlockAddrI);
    BlockAddrI->eraseFromParent();
  }
}

static bool isImplicitFallthrough(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return true;

  // Branching SPIR-V intrinsics are not detected by this generic method.
  // Thus, we can only trust negative result.
  if (!MBB.canFallThrough())
    return false;

  // Otherwise, we must manually check if we have a SPIR-V intrinsic which
  // prevent an implicit fallthrough.
  for (MachineBasicBlock::reverse_iterator It = MBB.rbegin(), E = MBB.rend();
       It != E; ++It) {
    if (isSpvIntrinsic(*It, Intrinsic::spv_switch))
      return false;
  }
  return true;
}

static void removeImplicitFallthroughs(MachineFunction &MF,
                                       MachineIRBuilder MIB) {
  // It is valid for MachineBasicBlocks to not finish with a branch instruction.
  // In such cases, they will simply fallthrough their immediate successor.
  for (MachineBasicBlock &MBB : MF) {
    if (!isImplicitFallthrough(MBB))
      continue;

    assert(std::distance(MBB.successors().begin(), MBB.successors().end()) ==
           1);
    MIB.setInsertPt(MBB, MBB.end());
    MIB.buildBr(**MBB.successors().begin());
  }
}

bool SPIRVPreLegalizer::runOnMachineFunction(MachineFunction &MF) {
  // Initialize the type registry.
  const SPIRVSubtarget &ST = MF.getSubtarget<SPIRVSubtarget>();
  SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  GR->setCurrentFunc(MF);
  MachineIRBuilder MIB(MF);
  // a registry of target extension constants
  DenseMap<MachineInstr *, Type *> TargetExtConstTypes;
  // to keep record of tracked constants
  addConstantsToTrack(MF, GR, ST, TargetExtConstTypes);
  foldConstantsIntoIntrinsics(MF, GR, MIB);
  insertBitcasts(MF, GR, MIB);
  generateAssignInstrs(MF, GR, MIB, TargetExtConstTypes);

  processSwitchesConstants(MF, GR, MIB);
  processBlockAddr(MF, GR, MIB);
  cleanupHelperInstructions(MF, GR);

  processInstrsWithTypeFolding(MF, GR, MIB);
  removeImplicitFallthroughs(MF, MIB);
  insertSpirvDecorations(MF, GR, MIB);
  insertInlineAsm(MF, GR, ST, MIB);
  selectOpBitcasts(MF, GR, MIB);

  return true;
}

INITIALIZE_PASS(SPIRVPreLegalizer, DEBUG_TYPE, "SPIRV pre legalizer", false,
                false)

char SPIRVPreLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPreLegalizerPass() {
  return new SPIRVPreLegalizer();
}
