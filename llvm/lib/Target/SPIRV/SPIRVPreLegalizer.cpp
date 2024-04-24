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
            for (unsigned i = 0; i < ConstVec->getNumElements(); ++i) {
              // Ensure that OpConstantComposite reuses a constant when it's
              // already created and available in the same machine function.
              Constant *ElemConst = ConstVec->getElementAsConstant(i);
              Register ElemReg = GR->find(ElemConst, &MF);
              if (!ElemReg.isValid())
                GR->add(ElemConst, &MF, BuildVec->getOperand(1 + i).getReg());
              else
                BuildVec->getOperand(1 + i).setReg(ElemReg);
            }
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
        const GlobalValue *Global = MI->getOperand(1).getGlobal();
        Type *ElementTy = GR->getDeducedGlobalValueType(Global);
        auto *Ty = TypedPointerType::get(ElementTy,
                                         Global->getType()->getAddressSpace());
        SpirvTy = GR->getOrCreateSPIRVType(Ty, MIB);
        break;
      }
      case TargetOpcode::G_ZEXT: {
        if (MI->getOperand(1).isReg()) {
          if (MachineInstr *DefInstr =
                  MRI.getVRegDef(MI->getOperand(1).getReg())) {
            if (SPIRVType *Def = propagateSPIRVType(DefInstr, GR, MRI, MIB)) {
              unsigned CurrentBW = GR->getScalarOrVectorBitWidth(Def);
              unsigned ExpectedBW =
                  std::max(MRI.getType(Reg).getScalarSizeInBits(), CurrentBW);
              unsigned NumElements = GR->getScalarOrVectorComponentCount(Def);
              SpirvTy = GR->getOrCreateSPIRVIntegerType(ExpectedBW, MIB);
              if (NumElements > 1)
                SpirvTy =
                    GR->getOrCreateSPIRVVectorType(SpirvTy, NumElements, MIB);
            }
          }
        }
        break;
      }
      case TargetOpcode::G_PTRTOINT:
        SpirvTy = GR->getOrCreateSPIRVIntegerType(
            MRI.getType(Reg).getScalarSizeInBits(), MIB);
        break;
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
createNewIdReg(SPIRVType *SpvType, Register SrcReg, MachineRegisterInfo &MRI,
               const SPIRVGlobalRegistry &GR) {
  if (!SpvType)
    SpvType = GR.getSPIRVTypeForVReg(SrcReg);
  assert(SpvType && "VReg is expected to have SPIRV type");
  LLT SrcLLT = MRI.getType(SrcReg);
  LLT NewT = LLT::scalar(32);
  bool IsFloat = SpvType->getOpcode() == SPIRV::OpTypeFloat;
  bool IsVectorFloat =
      SpvType->getOpcode() == SPIRV::OpTypeVector &&
      GR.getSPIRVTypeForVReg(SpvType->getOperand(1).getReg())->getOpcode() ==
          SPIRV::OpTypeFloat;
  IsFloat |= IsVectorFloat;
  auto GetIdOp = IsFloat ? SPIRV::GET_fID : SPIRV::GET_ID;
  auto DstClass = IsFloat ? &SPIRV::fIDRegClass : &SPIRV::IDRegClass;
  if (SrcLLT.isPointer()) {
    unsigned PtrSz = GR.getPointerSize();
    NewT = LLT::pointer(0, PtrSz);
    bool IsVec = SrcLLT.isVector();
    if (IsVec)
      NewT = LLT::fixed_vector(2, NewT);
    if (PtrSz == 64) {
      if (IsVec) {
        GetIdOp = SPIRV::GET_vpID64;
        DstClass = &SPIRV::vpID64RegClass;
      } else {
        GetIdOp = SPIRV::GET_pID64;
        DstClass = &SPIRV::pID64RegClass;
      }
    } else {
      if (IsVec) {
        GetIdOp = SPIRV::GET_vpID32;
        DstClass = &SPIRV::vpID32RegClass;
      } else {
        GetIdOp = SPIRV::GET_pID32;
        DstClass = &SPIRV::pID32RegClass;
      }
    }
  } else if (SrcLLT.isVector()) {
    NewT = LLT::fixed_vector(2, NewT);
    if (IsFloat) {
      GetIdOp = SPIRV::GET_vfID;
      DstClass = &SPIRV::vfIDRegClass;
    } else {
      GetIdOp = SPIRV::GET_vID;
      DstClass = &SPIRV::vIDRegClass;
    }
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
  SpirvTy = SpirvTy ? SpirvTy : GR->getOrCreateSPIRVType(Ty, MIB);
  Register NewReg = MRI.createGenericVirtualRegister(MRI.getType(Reg));
  if (auto *RC = MRI.getRegClassOrNull(Reg)) {
    MRI.setRegClass(NewReg, RC);
  } else {
    MRI.setRegClass(NewReg, &SPIRV::IDRegClass);
    MRI.setRegClass(Reg, &SPIRV::IDRegClass);
  }
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
  assert(MI.getNumDefs() > 0 && MRI.hasOneUse(MI.getOperand(0).getReg()));
  MachineInstr &AssignTypeInst =
      *(MRI.use_instr_begin(MI.getOperand(0).getReg()));
  auto NewReg =
      createNewIdReg(nullptr, MI.getOperand(0).getReg(), MRI, *GR).first;
  AssignTypeInst.getOperand(1).setReg(NewReg);
  MI.getOperand(0).setReg(NewReg);
  MIB.setInsertPt(*MI.getParent(),
                  (MI.getNextNode() ? MI.getNextNode()->getIterator()
                                    : MI.getParent()->end()));
  for (auto &Op : MI.operands()) {
    if (!Op.isReg() || Op.isDef())
      continue;
    auto IdOpInfo = createNewIdReg(nullptr, Op.getReg(), MRI, *GR);
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
                 MI.getOpcode() == TargetOpcode::G_ZEXT ||
                 MI.getOpcode() == TargetOpcode::G_PTRTOINT ||
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
      bool IsDstPtr = MRI.getType(DstReg).isPointer();
      bool isDstVec = MRI.getType(DstReg).isVector();
      if (IsDstPtr || isDstVec)
        MRI.setRegClass(DstReg, &SPIRV::IDRegClass);
      // Don't need to reset type of register holding constant and used in
      // G_ADDRSPACE_CAST, since it breaks legalizer.
      if (Opcode == TargetOpcode::G_CONSTANT && MRI.hasOneUse(DstReg)) {
        MachineInstr &UseMI = *MRI.use_instr_begin(DstReg);
        if (UseMI.getOpcode() == TargetOpcode::G_ADDRSPACE_CAST)
          continue;
      }
      MRI.setType(DstReg, IsDstPtr ? LLT::pointer(0, GR->getPointerSize())
                                   : LLT::scalar(32));
    }
  }
}

// Find basic blocks of the switch and replace registers in spv_switch() by its
// MBB equivalent.
static void processSwitches(MachineFunction &MF, SPIRVGlobalRegistry *GR,
                            MachineIRBuilder MIB) {
  DenseMap<const BasicBlock *, MachineBasicBlock *> BB2MBB;
  SmallVector<std::pair<MachineInstr *, SmallVector<MachineInstr *, 8>>>
      Switches;
  for (MachineBasicBlock &MBB : MF) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    BB2MBB[MBB.getBasicBlock()] = &MBB;
    for (MachineInstr &MI : MBB) {
      if (!isSpvIntrinsic(MI, Intrinsic::spv_switch))
        continue;
      // Calls to spv_switch intrinsics representing IR switches.
      SmallVector<MachineInstr *, 8> NewOps;
      for (unsigned i = 2; i < MI.getNumOperands(); ++i) {
        Register Reg = MI.getOperand(i).getReg();
        if (i % 2 == 1) {
          MachineInstr *ConstInstr = getDefInstrMaybeConstant(Reg, &MRI);
          NewOps.push_back(ConstInstr);
        } else {
          MachineInstr *BuildMBB = MRI.getVRegDef(Reg);
          assert(BuildMBB &&
                 BuildMBB->getOpcode() == TargetOpcode::G_BLOCK_ADDR &&
                 BuildMBB->getOperand(1).isBlockAddress() &&
                 BuildMBB->getOperand(1).getBlockAddress());
          NewOps.push_back(BuildMBB);
        }
      }
      Switches.push_back(std::make_pair(&MI, NewOps));
    }
  }

  SmallPtrSet<MachineInstr *, 8> ToEraseMI;
  for (auto &SwIt : Switches) {
    MachineInstr &MI = *SwIt.first;
    SmallVector<MachineInstr *, 8> &Ins = SwIt.second;
    SmallVector<MachineOperand, 8> NewOps;
    for (unsigned i = 0; i < Ins.size(); ++i) {
      if (Ins[i]->getOpcode() == TargetOpcode::G_BLOCK_ADDR) {
        BasicBlock *CaseBB =
            Ins[i]->getOperand(1).getBlockAddress()->getBasicBlock();
        auto It = BB2MBB.find(CaseBB);
        if (It == BB2MBB.end())
          report_fatal_error("cannot find a machine basic block by a basic "
                             "block in a switch statement");
        NewOps.push_back(MachineOperand::CreateMBB(It->second));
        MI.getParent()->addSuccessor(It->second);
        ToEraseMI.insert(Ins[i]);
      } else {
        NewOps.push_back(
            MachineOperand::CreateCImm(Ins[i]->getOperand(1).getCImm()));
      }
    }
    for (unsigned i = MI.getNumOperands() - 1; i > 1; --i)
      MI.removeOperand(i);
    for (auto &MO : NewOps)
      MI.addOperand(MO);
    if (MachineInstr *Next = MI.getNextNode()) {
      if (isSpvIntrinsic(*Next, Intrinsic::spv_track_constant)) {
        ToEraseMI.insert(Next);
        Next = MI.getNextNode();
      }
      if (Next && Next->getOpcode() == TargetOpcode::G_BRINDIRECT)
        ToEraseMI.insert(Next);
    }
  }
  for (MachineInstr *BlockAddrI : ToEraseMI)
    BlockAddrI->eraseFromParent();
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
