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
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Target/TargetIntrinsicInfo.h"

#define DEBUG_TYPE "spirv-prelegalizer"

using namespace llvm;

namespace {
class SPIRVPreLegalizer : public MachineFunctionPass {
public:
  static char ID;
  SPIRVPreLegalizer() : MachineFunctionPass(ID) {
    initializeSPIRVPreLegalizerPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

static void addConstantsToTrack(MachineFunction &MF, SPIRVGlobalRegistry *GR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DenseMap<MachineInstr *, Register> RegsAlreadyAddedToDT;
  SmallVector<MachineInstr *, 10> ToErase, ToEraseComposites;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_track_constant))
        continue;
      ToErase.push_back(&MI);
      auto *Const =
          cast<Constant>(cast<ConstantAsMetadata>(
                             MI.getOperand(3).getMetadata()->getOperand(0))
                             ->getValue());
      if (auto *GV = dyn_cast<GlobalValue>(Const)) {
        Register Reg = GR->find(GV, &MF);
        if (!Reg.isValid())
          GR->add(GV, &MF, MI.getOperand(2).getReg());
        else
          RegsAlreadyAddedToDT[&MI] = Reg;
      } else {
        Register Reg = GR->find(Const, &MF);
        if (!Reg.isValid()) {
          if (auto *ConstVec = dyn_cast<ConstantDataVector>(Const)) {
            auto *BuildVec = MRI.getVRegDef(MI.getOperand(2).getReg());
            assert(BuildVec &&
                   BuildVec->getOpcode() == TargetOpcode::G_BUILD_VECTOR);
            for (unsigned i = 0; i < ConstVec->getNumElements(); ++i)
              GR->add(ConstVec->getElementAsConstant(i), &MF,
                      BuildVec->getOperand(1 + i).getReg());
          }
          GR->add(Const, &MF, MI.getOperand(2).getReg());
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
    if (RegsAlreadyAddedToDT.contains(MI))
      Reg = RegsAlreadyAddedToDT[MI];
    auto *RC = MRI.getRegClassOrNull(MI->getOperand(0).getReg());
    if (!MRI.getRegClassOrNull(Reg) && RC)
      MRI.setRegClass(Reg, RC);
    MRI.replaceRegWith(MI->getOperand(0).getReg(), Reg);
    MI->eraseFromParent();
  }
  for (MachineInstr *MI : ToEraseComposites)
    MI->eraseFromParent();
}

static void foldConstantsIntoIntrinsics(MachineFunction &MF) {
  SmallVector<MachineInstr *, 10> ToErase;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned AssignNameOperandShift = 2;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_assign_name))
        continue;
      unsigned NumOp = MI.getNumExplicitDefs() + AssignNameOperandShift;
      while (MI.getOperand(NumOp).isReg()) {
        MachineOperand &MOp = MI.getOperand(NumOp);
        MachineInstr *ConstMI = MRI.getVRegDef(MOp.getReg());
        assert(ConstMI->getOpcode() == TargetOpcode::G_CONSTANT);
        MI.removeOperand(NumOp);
        MI.addOperand(MachineOperand::CreateImm(
            ConstMI->getOperand(1).getCImm()->getZExtValue()));
        if (MRI.use_empty(ConstMI->getOperand(0).getReg()))
          ToErase.push_back(ConstMI);
      }
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
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
      SPIRVType *BaseTy = GR->getOrCreateSPIRVType(
          getMDOperandAsType(MI.getOperand(3).getMetadata(), 0), MIB);
      SPIRVType *AssignedPtrType = GR->getOrCreateSPIRVPointerType(
          BaseTy, MI, *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo(),
          addressSpaceToStorageClass(MI.getOperand(4).getImm(), *ST));

      // If the bitcast would be redundant, replace all uses with the source
      // register.
      if (GR->getSPIRVTypeForVReg(Source) == AssignedPtrType) {
        MIB.getMRI()->replaceRegWith(Def, Source);
      } else {
        GR->assignSPIRVTypeToVReg(AssignedPtrType, Def, MF);
        MIB.buildBitcast(Def, Source);
      }
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
}

// Translating GV, IRTranslator sometimes generates following IR:
//   %1 = G_GLOBAL_VALUE
//   %2 = COPY %1
//   %3 = G_ADDRSPACE_CAST %2
// New registers have no SPIRVType and no register class info.
//
// Set SPIRVType for GV, propagate it from GV to other instructions,
// also set register classes.
static SPIRVType *propagateSPIRVType(MachineInstr *MI, SPIRVGlobalRegistry *GR,
                                     MachineRegisterInfo &MRI,
                                     MachineIRBuilder &MIB) {
  SPIRVType *SpirvTy = nullptr;
  assert(MI && "Machine instr is expected");
  if (MI->getOperand(0).isReg()) {
    Register Reg = MI->getOperand(0).getReg();
    SpirvTy = GR->getSPIRVTypeForVReg(Reg);
    if (!SpirvTy) {
      switch (MI->getOpcode()) {
      case TargetOpcode::G_CONSTANT: {
        MIB.setInsertPt(*MI->getParent(), MI);
        Type *Ty = MI->getOperand(1).getCImm()->getType();
        SpirvTy = GR->getOrCreateSPIRVType(Ty, MIB);
        break;
      }
      case TargetOpcode::G_GLOBAL_VALUE: {
        MIB.setInsertPt(*MI->getParent(), MI);
        Type *Ty = MI->getOperand(1).getGlobal()->getType();
        SpirvTy = GR->getOrCreateSPIRVType(Ty, MIB);
        break;
      }
      case TargetOpcode::G_TRUNC:
      case TargetOpcode::G_ADDRSPACE_CAST:
      case TargetOpcode::G_PTR_ADD:
      case TargetOpcode::COPY: {
        MachineOperand &Op = MI->getOperand(1);
        MachineInstr *Def = Op.isReg() ? MRI.getVRegDef(Op.getReg()) : nullptr;
        if (Def)
          SpirvTy = propagateSPIRVType(Def, GR, MRI, MIB);
        break;
      }
      default:
        break;
      }
      if (SpirvTy)
        GR->assignSPIRVTypeToVReg(SpirvTy, Reg, MIB.getMF());
      if (!MRI.getRegClassOrNull(Reg))
        MRI.setRegClass(Reg, &SPIRV::IDRegClass);
    }
  }
  return SpirvTy;
}

static std::pair<Register, unsigned>
createNewIdReg(Register ValReg, unsigned Opcode, MachineRegisterInfo &MRI,
               const SPIRVGlobalRegistry &GR) {
  LLT NewT = LLT::scalar(32);
  SPIRVType *SpvType = GR.getSPIRVTypeForVReg(ValReg);
  assert(SpvType && "VReg is expected to have SPIRV type");
  bool IsFloat = SpvType->getOpcode() == SPIRV::OpTypeFloat;
  bool IsVectorFloat =
      SpvType->getOpcode() == SPIRV::OpTypeVector &&
      GR.getSPIRVTypeForVReg(SpvType->getOperand(1).getReg())->getOpcode() ==
          SPIRV::OpTypeFloat;
  IsFloat |= IsVectorFloat;
  auto GetIdOp = IsFloat ? SPIRV::GET_fID : SPIRV::GET_ID;
  auto DstClass = IsFloat ? &SPIRV::fIDRegClass : &SPIRV::IDRegClass;
  if (MRI.getType(ValReg).isPointer()) {
    NewT = LLT::pointer(0, 32);
    GetIdOp = SPIRV::GET_pID;
    DstClass = &SPIRV::pIDRegClass;
  } else if (MRI.getType(ValReg).isVector()) {
    NewT = LLT::fixed_vector(2, NewT);
    GetIdOp = IsFloat ? SPIRV::GET_vfID : SPIRV::GET_vID;
    DstClass = IsFloat ? &SPIRV::vfIDRegClass : &SPIRV::vIDRegClass;
  }
  Register IdReg = MRI.createGenericVirtualRegister(NewT);
  MRI.setRegClass(IdReg, DstClass);
  return {IdReg, GetIdOp};
}

// Insert ASSIGN_TYPE instuction between Reg and its definition, set NewReg as
// a dst of the definition, assign SPIRVType to both registers. If SpirvTy is
// provided, use it as SPIRVType in ASSIGN_TYPE, otherwise create it from Ty.
// It's used also in SPIRVBuiltins.cpp.
// TODO: maybe move to SPIRVUtils.
namespace llvm {
Register insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                           SPIRVGlobalRegistry *GR, MachineIRBuilder &MIB,
                           MachineRegisterInfo &MRI) {
  MachineInstr *Def = MRI.getVRegDef(Reg);
  assert((Ty || SpirvTy) && "Either LLVM or SPIRV type is expected.");
  MIB.setInsertPt(*Def->getParent(),
                  (Def->getNextNode() ? Def->getNextNode()->getIterator()
                                      : Def->getParent()->end()));
  Register NewReg = MRI.createGenericVirtualRegister(MRI.getType(Reg));
  if (auto *RC = MRI.getRegClassOrNull(Reg)) {
    MRI.setRegClass(NewReg, RC);
  } else {
    MRI.setRegClass(NewReg, &SPIRV::IDRegClass);
    MRI.setRegClass(Reg, &SPIRV::IDRegClass);
  }
  SpirvTy = SpirvTy ? SpirvTy : GR->getOrCreateSPIRVType(Ty, MIB);
  GR->assignSPIRVTypeToVReg(SpirvTy, Reg, MIB.getMF());
  // This is to make it convenient for Legalizer to get the SPIRVType
  // when processing the actual MI (i.e. not pseudo one).
  GR->assignSPIRVTypeToVReg(SpirvTy, NewReg, MIB.getMF());
  // Copy MIFlags from Def to ASSIGN_TYPE instruction. It's required to keep
  // the flags after instruction selection.
  const uint32_t Flags = Def->getFlags();
  MIB.buildInstr(SPIRV::ASSIGN_TYPE)
      .addDef(Reg)
      .addUse(NewReg)
      .addUse(GR->getSPIRVTypeID(SpirvTy))
      .setMIFlags(Flags);
  Def->getOperand(0).setReg(NewReg);
  return NewReg;
}

void processInstr(MachineInstr &MI, MachineIRBuilder &MIB,
                  MachineRegisterInfo &MRI, SPIRVGlobalRegistry *GR) {
  unsigned Opc = MI.getOpcode();
  assert(MI.getNumDefs() > 0 && MRI.hasOneUse(MI.getOperand(0).getReg()));
  MachineInstr &AssignTypeInst =
      *(MRI.use_instr_begin(MI.getOperand(0).getReg()));
  auto NewReg = createNewIdReg(MI.getOperand(0).getReg(), Opc, MRI, *GR).first;
  AssignTypeInst.getOperand(1).setReg(NewReg);
  MI.getOperand(0).setReg(NewReg);
  MIB.setInsertPt(*MI.getParent(),
                  (MI.getNextNode() ? MI.getNextNode()->getIterator()
                                    : MI.getParent()->end()));
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || Op.isDef())
      continue;
    auto IdOpInfo = createNewIdReg(Op.getReg(), Opc, MRI, *GR);
    MIB.buildInstr(IdOpInfo.second).addDef(IdOpInfo.first).addUse(Op.getReg());
    Op.setReg(IdOpInfo.first);
  }
}
} // namespace llvm

static void generateAssignInstrs(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                                 MachineIRBuilder MIB) {
  // Get access to information about available extensions
  const SPIRVSubtarget *ST =
      static_cast<const SPIRVSubtarget *>(&MIB.getMF().getSubtarget());

  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<MachineInstr *, 10> ToErase;

  for (MachineBasicBlock *MBB : post_order(&MF)) {
    if (MBB->empty())
      continue;

    bool ReachedBegin = false;
    for (auto MII = std::prev(MBB->end()), Begin = MBB->begin();
         !ReachedBegin;) {
      MachineInstr &MI = *MII;

      if (isSpvIntrinsic(MI, Intrinsic::spv_assign_ptr_type)) {
        Register Reg = MI.getOperand(1).getReg();
        MIB.setInsertPt(*MI.getParent(), MI.getIterator());
        SPIRVType *BaseTy = GR->getOrCreateSPIRVType(
            getMDOperandAsType(MI.getOperand(2).getMetadata(), 0), MIB);
        SPIRVType *AssignedPtrType = GR->getOrCreateSPIRVPointerType(
            BaseTy, MI, *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo(),
            addressSpaceToStorageClass(MI.getOperand(3).getImm(), *ST));
        MachineInstr *Def = MRI.getVRegDef(Reg);
        assert(Def && "Expecting an instruction that defines the register");
        // G_GLOBAL_VALUE already has type info.
        if (Def->getOpcode() != TargetOpcode::G_GLOBAL_VALUE)
          insertAssignInstr(Reg, nullptr, AssignedPtrType, GR, MIB,
                            MF.getRegInfo());
        ToErase.push_back(&MI);
      } else if (isSpvIntrinsic(MI, Intrinsic::spv_assign_type)) {
        Register Reg = MI.getOperand(1).getReg();
        Type *Ty = getMDOperandAsType(MI.getOperand(2).getMetadata(), 0);
        MachineInstr *Def = MRI.getVRegDef(Reg);
        assert(Def && "Expecting an instruction that defines the register");
        // G_GLOBAL_VALUE already has type info.
        if (Def->getOpcode() != TargetOpcode::G_GLOBAL_VALUE)
          insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MF.getRegInfo());
        ToErase.push_back(&MI);
      } else if (MI.getOpcode() == TargetOpcode::G_CONSTANT ||
                 MI.getOpcode() == TargetOpcode::G_FCONSTANT ||
                 MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
        // %rc = G_CONSTANT ty Val
        // ===>
        // %cty = OpType* ty
        // %rctmp = G_CONSTANT ty Val
        // %rc = ASSIGN_TYPE %rctmp, %cty
        Register Reg = MI.getOperand(0).getReg();
        if (MRI.hasOneUse(Reg)) {
          MachineInstr &UseMI = *MRI.use_instr_begin(Reg);
          if (isSpvIntrinsic(UseMI, Intrinsic::spv_assign_type) ||
              isSpvIntrinsic(UseMI, Intrinsic::spv_assign_name))
            continue;
        }
        Type *Ty = nullptr;
        if (MI.getOpcode() == TargetOpcode::G_CONSTANT)
          Ty = MI.getOperand(1).getCImm()->getType();
        else if (MI.getOpcode() == TargetOpcode::G_FCONSTANT)
          Ty = MI.getOperand(1).getFPImm()->getType();
        else {
          assert(MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR);
          Type *ElemTy = nullptr;
          MachineInstr *ElemMI = MRI.getVRegDef(MI.getOperand(1).getReg());
          assert(ElemMI);

          if (ElemMI->getOpcode() == TargetOpcode::G_CONSTANT)
            ElemTy = ElemMI->getOperand(1).getCImm()->getType();
          else if (ElemMI->getOpcode() == TargetOpcode::G_FCONSTANT)
            ElemTy = ElemMI->getOperand(1).getFPImm()->getType();
          else
            llvm_unreachable("Unexpected opcode");
          unsigned NumElts =
              MI.getNumExplicitOperands() - MI.getNumExplicitDefs();
          Ty = VectorType::get(ElemTy, NumElts, false);
        }
        insertAssignInstr(Reg, Ty, nullptr, GR, MIB, MRI);
      } else if (MI.getOpcode() == TargetOpcode::G_TRUNC ||
                 MI.getOpcode() == TargetOpcode::G_GLOBAL_VALUE ||
                 MI.getOpcode() == TargetOpcode::COPY ||
                 MI.getOpcode() == TargetOpcode::G_ADDRSPACE_CAST) {
        propagateSPIRVType(&MI, GR, MRI, MIB);
      }

      if (MII == Begin)
        ReachedBegin = true;
      else
        --MII;
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
}

// Defined in SPIRVLegalizerInfo.cpp.
extern bool isTypeFoldingSupported(unsigned Opcode);

static void processInstrsWithTypeFolding(MachineFunction &MF,
                                         SPIRVGlobalRegistry *GR,
                                         MachineIRBuilder MIB) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (isTypeFoldingSupported(MI.getOpcode()))
        processInstr(MI, MIB, MRI, GR);
    }
  }
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // We need to rewrite dst types for ASSIGN_TYPE instrs to be able
      // to perform tblgen'erated selection and we can't do that on Legalizer
      // as it operates on gMIR only.
      if (MI.getOpcode() != SPIRV::ASSIGN_TYPE)
        continue;
      Register SrcReg = MI.getOperand(1).getReg();
      unsigned Opcode = MRI.getVRegDef(SrcReg)->getOpcode();
      if (!isTypeFoldingSupported(Opcode))
        continue;
      Register DstReg = MI.getOperand(0).getReg();
      if (MRI.getType(DstReg).isVector())
        MRI.setRegClass(DstReg, &SPIRV::IDRegClass);
      // Don't need to reset type of register holding constant and used in
      // G_ADDRSPACE_CAST, since it braaks legalizer.
      if (Opcode == TargetOpcode::G_CONSTANT && MRI.hasOneUse(DstReg)) {
        MachineInstr &UseMI = *MRI.use_instr_begin(DstReg);
        if (UseMI.getOpcode() == TargetOpcode::G_ADDRSPACE_CAST)
          continue;
      }
      MRI.setType(DstReg, LLT::scalar(32));
    }
  }
}

static void processSwitches(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                            MachineIRBuilder MIB) {
  // Before IRTranslator pass, calls to spv_switch intrinsic are inserted before
  // each switch instruction. IRTranslator lowers switches to G_ICMP + G_BRCOND
  // + G_BR triples. A switch with two cases may be transformed to this MIR
  // sequence:
  //
  //   intrinsic(@llvm.spv.switch), %CmpReg, %Const0, %Const1
  //   %Dst0 = G_ICMP intpred(eq), %CmpReg, %Const0
  //   G_BRCOND %Dst0, %bb.2
  //   G_BR %bb.5
  // bb.5.entry:
  //   %Dst1 = G_ICMP intpred(eq), %CmpReg, %Const1
  //   G_BRCOND %Dst1, %bb.3
  //   G_BR %bb.4
  // bb.2.sw.bb:
  //   ...
  // bb.3.sw.bb1:
  //   ...
  // bb.4.sw.epilog:
  //   ...
  //
  // Sometimes (in case of range-compare switches), additional G_SUBs
  // instructions are inserted before G_ICMPs. Those need to be additionally
  // processed.
  //
  // This function modifies spv_switch call's operands to include destination
  // MBBs (default and for each constant value).
  //
  // At the end, the function removes redundant [G_SUB] + G_ICMP + G_BRCOND +
  // G_BR sequences.

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Collect spv_switches and G_ICMPs across all MBBs in MF.
  std::vector<MachineInstr *> RelevantInsts;

  // Collect redundant MIs from [G_SUB] + G_ICMP + G_BRCOND + G_BR sequences.
  // After updating spv_switches, the instructions can be removed.
  std::vector<MachineInstr *> PostUpdateArtifacts;

  // Temporary set of compare registers. G_SUBs and G_ICMPs relating to
  // spv_switch use these registers.
  DenseSet<Register> CompareRegs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Calls to spv_switch intrinsics representing IR switches.
      if (isSpvIntrinsic(MI, Intrinsic::spv_switch)) {
        assert(MI.getOperand(1).isReg());
        CompareRegs.insert(MI.getOperand(1).getReg());
        RelevantInsts.push_back(&MI);
      }

      // G_SUBs coming from range-compare switch lowering. G_SUBs are found
      // after spv_switch but before G_ICMP.
      if (MI.getOpcode() == TargetOpcode::G_SUB && MI.getOperand(1).isReg() &&
          CompareRegs.contains(MI.getOperand(1).getReg())) {
        assert(MI.getOperand(0).isReg() && MI.getOperand(1).isReg());
        Register Dst = MI.getOperand(0).getReg();
        CompareRegs.insert(Dst);
        PostUpdateArtifacts.push_back(&MI);
      }

      // G_ICMPs relating to switches.
      if (MI.getOpcode() == TargetOpcode::G_ICMP && MI.getOperand(2).isReg() &&
          CompareRegs.contains(MI.getOperand(2).getReg())) {
        Register Dst = MI.getOperand(0).getReg();
        RelevantInsts.push_back(&MI);
        PostUpdateArtifacts.push_back(&MI);
        MachineInstr *CBr = MRI.use_begin(Dst)->getParent();
        assert(CBr->getOpcode() == SPIRV::G_BRCOND);
        PostUpdateArtifacts.push_back(CBr);
        MachineInstr *Br = CBr->getNextNode();
        assert(Br->getOpcode() == SPIRV::G_BR);
        PostUpdateArtifacts.push_back(Br);
      }
    }
  }

  // Update each spv_switch with destination MBBs.
  for (auto i = RelevantInsts.begin(); i != RelevantInsts.end(); i++) {
    if (!isSpvIntrinsic(**i, Intrinsic::spv_switch))
      continue;

    // Currently considered spv_switch.
    MachineInstr *Switch = *i;
    // Set the first successor as default MBB to support empty switches.
    MachineBasicBlock *DefaultMBB = *Switch->getParent()->succ_begin();
    // Container for mapping values to MMBs.
    SmallDenseMap<uint64_t, MachineBasicBlock *> ValuesToMBBs;

    // Walk all G_ICMPs to collect ValuesToMBBs. Start at currently considered
    // spv_switch (i) and break at any spv_switch with the same compare
    // register (indicating we are back at the same scope).
    Register CompareReg = Switch->getOperand(1).getReg();
    for (auto j = i + 1; j != RelevantInsts.end(); j++) {
      if (isSpvIntrinsic(**j, Intrinsic::spv_switch) &&
          (*j)->getOperand(1).getReg() == CompareReg)
        break;

      if (!((*j)->getOpcode() == TargetOpcode::G_ICMP &&
            (*j)->getOperand(2).getReg() == CompareReg))
        continue;

      MachineInstr *ICMP = *j;
      Register Dst = ICMP->getOperand(0).getReg();
      MachineOperand &PredOp = ICMP->getOperand(1);
      const auto CC = static_cast<CmpInst::Predicate>(PredOp.getPredicate());
      assert((CC == CmpInst::ICMP_EQ || CC == CmpInst::ICMP_ULE) &&
             MRI.hasOneUse(Dst) && MRI.hasOneDef(CompareReg));
      uint64_t Value = getIConstVal(ICMP->getOperand(3).getReg(), &MRI);
      MachineInstr *CBr = MRI.use_begin(Dst)->getParent();
      assert(CBr->getOpcode() == SPIRV::G_BRCOND && CBr->getOperand(1).isMBB());
      MachineBasicBlock *MBB = CBr->getOperand(1).getMBB();

      // Map switch case Value to target MBB.
      ValuesToMBBs[Value] = MBB;

      // Add target MBB as successor to the switch's MBB.
      Switch->getParent()->addSuccessor(MBB);

      // The next MI is always G_BR to either the next case or the default.
      MachineInstr *NextMI = CBr->getNextNode();
      assert(NextMI->getOpcode() == SPIRV::G_BR &&
             NextMI->getOperand(0).isMBB());
      MachineBasicBlock *NextMBB = NextMI->getOperand(0).getMBB();
      // Default MBB does not begin with G_ICMP using spv_switch compare
      // register.
      if (NextMBB->front().getOpcode() != SPIRV::G_ICMP ||
          (NextMBB->front().getOperand(2).isReg() &&
           NextMBB->front().getOperand(2).getReg() != CompareReg)) {
        // Set default MBB and add it as successor to the switch's MBB.
        DefaultMBB = NextMBB;
        Switch->getParent()->addSuccessor(DefaultMBB);
      }
    }

    // Modify considered spv_switch operands using collected Values and
    // MBBs.
    SmallVector<const ConstantInt *, 3> Values;
    SmallVector<MachineBasicBlock *, 3> MBBs;
    for (unsigned k = 2; k < Switch->getNumExplicitOperands(); k++) {
      Register CReg = Switch->getOperand(k).getReg();
      uint64_t Val = getIConstVal(CReg, &MRI);
      MachineInstr *ConstInstr = getDefInstrMaybeConstant(CReg, &MRI);
      if (!ValuesToMBBs[Val])
        continue;

      Values.push_back(ConstInstr->getOperand(1).getCImm());
      MBBs.push_back(ValuesToMBBs[Val]);
    }

    for (unsigned k = Switch->getNumExplicitOperands() - 1; k > 1; k--)
      Switch->removeOperand(k);

    Switch->addOperand(MachineOperand::CreateMBB(DefaultMBB));
    for (unsigned k = 0; k < Values.size(); k++) {
      Switch->addOperand(MachineOperand::CreateCImm(Values[k]));
      Switch->addOperand(MachineOperand::CreateMBB(MBBs[k]));
    }
  }

  for (MachineInstr *MI : PostUpdateArtifacts) {
    MachineBasicBlock *ParentMBB = MI->getParent();
    MI->eraseFromParent();
    // If G_ICMP + G_BRCOND + G_BR were the only MIs in MBB, erase this MBB. It
    // can be safely assumed, there are no breaks or phis directing into this
    // MBB. However, we need to remove this MBB from the CFG graph. MBBs must be
    // erased top-down.
    if (ParentMBB->empty()) {
      while (!ParentMBB->pred_empty())
        (*ParentMBB->pred_begin())->removeSuccessor(ParentMBB);

      while (!ParentMBB->succ_empty())
        ParentMBB->removeSuccessor(ParentMBB->succ_begin());

      ParentMBB->eraseFromParent();
    }
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
  addConstantsToTrack(MF, GR);
  foldConstantsIntoIntrinsics(MF);
  insertBitcasts(MF, GR, MIB);
  generateAssignInstrs(MF, GR, MIB);
  processSwitches(MF, GR, MIB);
  processInstrsWithTypeFolding(MF, GR, MIB);
  removeImplicitFallthroughs(MF, MIB);

  return true;
}

INITIALIZE_PASS(SPIRVPreLegalizer, DEBUG_TYPE, "SPIRV pre legalizer", false,
                false)

char SPIRVPreLegalizer::ID = 0;

FunctionPass *llvm::createSPIRVPreLegalizerPass() {
  return new SPIRVPreLegalizer();
}
