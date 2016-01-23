//===-- R600ISelLowering.cpp - R600 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Custom DAG lowering for R600
//
//===----------------------------------------------------------------------===//

#include "R600ISelLowering.h"
#include "AMDGPUFrameLowering.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"

using namespace llvm;

R600TargetLowering::R600TargetLowering(TargetMachine &TM,
                                       const AMDGPUSubtarget &STI)
    : AMDGPUTargetLowering(TM, STI), Gen(STI.getGeneration()) {
  addRegisterClass(MVT::v4f32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::R600_Reg32RegClass);
  addRegisterClass(MVT::v4i32, &AMDGPU::R600_Reg128RegClass);
  addRegisterClass(MVT::i32, &AMDGPU::R600_Reg32RegClass);
  addRegisterClass(MVT::v2f32, &AMDGPU::R600_Reg64RegClass);
  addRegisterClass(MVT::v2i32, &AMDGPU::R600_Reg64RegClass);

  computeRegisterProperties(STI.getRegisterInfo());

  // Set condition code actions
  setCondCodeAction(ISD::SETO,   MVT::f32, Expand);
  setCondCodeAction(ISD::SETUO,  MVT::f32, Expand);
  setCondCodeAction(ISD::SETLT,  MVT::f32, Expand);
  setCondCodeAction(ISD::SETLE,  MVT::f32, Expand);
  setCondCodeAction(ISD::SETOLT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETOLE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETONE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUEQ, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUGE, MVT::f32, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETULT, MVT::f32, Expand);
  setCondCodeAction(ISD::SETULE, MVT::f32, Expand);

  setCondCodeAction(ISD::SETLE, MVT::i32, Expand);
  setCondCodeAction(ISD::SETLT, MVT::i32, Expand);
  setCondCodeAction(ISD::SETULE, MVT::i32, Expand);
  setCondCodeAction(ISD::SETULT, MVT::i32, Expand);

  setOperationAction(ISD::FCOS, MVT::f32, Custom);
  setOperationAction(ISD::FSIN, MVT::f32, Custom);

  setOperationAction(ISD::SETCC, MVT::v4i32, Expand);
  setOperationAction(ISD::SETCC, MVT::v2i32, Expand);

  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);

  setOperationAction(ISD::FSUB, MVT::f32, Expand);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i1, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i1, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::v2i32, Expand);
  setOperationAction(ISD::SELECT, MVT::v4i32, Expand);

  // ADD, SUB overflow.
  // TODO: turn these into Legal?
  if (Subtarget->hasCARRY())
    setOperationAction(ISD::UADDO, MVT::i32, Custom);

  if (Subtarget->hasBORROW())
    setOperationAction(ISD::USUBO, MVT::i32, Custom);

  // Expand sign extension of vectors
  if (!Subtarget->hasBFE())
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Expand);

  if (!Subtarget->hasBFE())
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Expand);

  if (!Subtarget->hasBFE())
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Legal);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i32, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Expand);


  // Legalize loads and stores to the private address space.
  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);

  // EXTLOAD should be the same as ZEXTLOAD. It is legal for some address
  // spaces, so it is custom lowered to handle those where it isn't.
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Custom);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i16, Custom);

    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i8, Custom);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i16, Custom);

    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i8, Custom);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i16, Custom);
  }

  setOperationAction(ISD::STORE, MVT::i8, Custom);
  setOperationAction(ISD::STORE, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::v2i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);
  setTruncStoreAction(MVT::i32, MVT::i8, Custom);
  setTruncStoreAction(MVT::i32, MVT::i16, Custom);

  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i32, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f32, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i32, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f32, Custom);

  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i32, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f32, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i32, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4f32, Custom);

  setTargetDAGCombine(ISD::FP_ROUND);
  setTargetDAGCombine(ISD::FP_TO_SINT);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::SELECT_CC);
  setTargetDAGCombine(ISD::INSERT_VECTOR_ELT);

  // We don't have 64-bit shifts. Thus we need either SHX i64 or SHX_PARTS i32
  //  to be Legal/Custom in order to avoid library calls.
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);

  const MVT ScalarIntVTs[] = { MVT::i32, MVT::i64 };
  for (MVT VT : ScalarIntVTs) {
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::SUBE, VT, Expand);
  }

  setSchedulingPreference(Sched::Source);
}

static inline bool isEOP(MachineBasicBlock::iterator I) {
  return std::next(I)->getOpcode() == AMDGPU::RETURN;
}

MachineBasicBlock * R600TargetLowering::EmitInstrWithCustomInserter(
    MachineInstr * MI, MachineBasicBlock * BB) const {
  MachineFunction * MF = BB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  MachineBasicBlock::iterator I = *MI;
  const R600InstrInfo *TII =
      static_cast<const R600InstrInfo *>(Subtarget->getInstrInfo());

  switch (MI->getOpcode()) {
  default:
    // Replace LDS_*_RET instruction that don't have any uses with the
    // equivalent LDS_*_NORET instruction.
    if (TII->isLDSRetInstr(MI->getOpcode())) {
      int DstIdx = TII->getOperandIdx(MI->getOpcode(), AMDGPU::OpName::dst);
      assert(DstIdx != -1);
      MachineInstrBuilder NewMI;
      // FIXME: getLDSNoRetOp method only handles LDS_1A1D LDS ops. Add
      //        LDS_1A2D support and remove this special case.
      if (!MRI.use_empty(MI->getOperand(DstIdx).getReg()) ||
           MI->getOpcode() == AMDGPU::LDS_CMPST_RET)
        return BB;

      NewMI = BuildMI(*BB, I, BB->findDebugLoc(I),
                      TII->get(AMDGPU::getLDSNoRetOp(MI->getOpcode())));
      for (unsigned i = 1, e = MI->getNumOperands(); i < e; ++i) {
        NewMI.addOperand(MI->getOperand(i));
      }
    } else {
      return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
    }
    break;
  case AMDGPU::CLAMP_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                   AMDGPU::MOV,
                                                   MI->getOperand(0).getReg(),
                                                   MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_CLAMP);
    break;
  }

  case AMDGPU::FABS_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                    AMDGPU::MOV,
                                                    MI->getOperand(0).getReg(),
                                                    MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_ABS);
    break;
  }

  case AMDGPU::FNEG_R600: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, I,
                                                    AMDGPU::MOV,
                                                    MI->getOperand(0).getReg(),
                                                    MI->getOperand(1).getReg());
    TII->addFlag(NewMI, 0, MO_FLAG_NEG);
    break;
  }

  case AMDGPU::MASK_WRITE: {
    unsigned maskedRegister = MI->getOperand(0).getReg();
    assert(TargetRegisterInfo::isVirtualRegister(maskedRegister));
    MachineInstr * defInstr = MRI.getVRegDef(maskedRegister);
    TII->addFlag(defInstr, 0, MO_FLAG_MASK);
    break;
  }

  case AMDGPU::MOV_IMM_F32:
    TII->buildMovImm(*BB, I, MI->getOperand(0).getReg(),
                     MI->getOperand(1).getFPImm()->getValueAPF()
                         .bitcastToAPInt().getZExtValue());
    break;
  case AMDGPU::MOV_IMM_I32:
    TII->buildMovImm(*BB, I, MI->getOperand(0).getReg(),
                     MI->getOperand(1).getImm());
    break;
  case AMDGPU::CONST_COPY: {
    MachineInstr *NewMI = TII->buildDefaultInstruction(*BB, MI, AMDGPU::MOV,
        MI->getOperand(0).getReg(), AMDGPU::ALU_CONST);
    TII->setImmOperand(NewMI, AMDGPU::OpName::src0_sel,
        MI->getOperand(1).getImm());
    break;
  }

  case AMDGPU::RAT_WRITE_CACHELESS_32_eg:
  case AMDGPU::RAT_WRITE_CACHELESS_64_eg:
  case AMDGPU::RAT_WRITE_CACHELESS_128_eg: {
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(isEOP(I)); // Set End of program bit
    break;
  }
  case AMDGPU::RAT_STORE_TYPED_eg: {
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addOperand(MI->getOperand(2))
            .addImm(isEOP(I)); // Set End of program bit
    break;
  }

  case AMDGPU::TXD: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    MachineOperand &RID = MI->getOperand(4);
    MachineOperand &SID = MI->getOperand(5);
    unsigned TextureId = MI->getOperand(6).getImm();
    unsigned SrcX = 0, SrcY = 1, SrcZ = 2, SrcW = 3;
    unsigned CTX = 1, CTY = 1, CTZ = 1, CTW = 1;

    switch (TextureId) {
    case 5: // Rect
      CTX = CTY = 0;
      break;
    case 6: // Shadow1D
      SrcW = SrcZ;
      break;
    case 7: // Shadow2D
      SrcW = SrcZ;
      break;
    case 8: // ShadowRect
      CTX = CTY = 0;
      SrcW = SrcZ;
      break;
    case 9: // 1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 10: // 2DArray
      CTZ = 0;
      break;
    case 11: // Shadow1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 12: // Shadow2DArray
      CTZ = 0;
      break;
    }
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW)
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::TXD_SHADOW: {
    unsigned T0 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    unsigned T1 = MRI.createVirtualRegister(&AMDGPU::R600_Reg128RegClass);
    MachineOperand &RID = MI->getOperand(4);
    MachineOperand &SID = MI->getOperand(5);
    unsigned TextureId = MI->getOperand(6).getImm();
    unsigned SrcX = 0, SrcY = 1, SrcZ = 2, SrcW = 3;
    unsigned CTX = 1, CTY = 1, CTZ = 1, CTW = 1;

    switch (TextureId) {
    case 5: // Rect
      CTX = CTY = 0;
      break;
    case 6: // Shadow1D
      SrcW = SrcZ;
      break;
    case 7: // Shadow2D
      SrcW = SrcZ;
      break;
    case 8: // ShadowRect
      CTX = CTY = 0;
      SrcW = SrcZ;
      break;
    case 9: // 1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 10: // 2DArray
      CTZ = 0;
      break;
    case 11: // Shadow1DArray
      SrcZ = SrcY;
      CTZ = 0;
      break;
    case 12: // Shadow2DArray
      CTZ = 0;
      break;
    }

    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_H), T0)
            .addOperand(MI->getOperand(3))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SET_GRADIENTS_V), T1)
            .addOperand(MI->getOperand(2))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::TEX_SAMPLE_C_G))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addImm(SrcX)
            .addImm(SrcY)
            .addImm(SrcZ)
            .addImm(SrcW)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(0)
            .addImm(1)
            .addImm(2)
            .addImm(3)
            .addOperand(RID)
            .addOperand(SID)
            .addImm(CTX)
            .addImm(CTY)
            .addImm(CTZ)
            .addImm(CTW)
            .addReg(T0, RegState::Implicit)
            .addReg(T1, RegState::Implicit);
    break;
  }

  case AMDGPU::BRANCH:
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP))
              .addOperand(MI->getOperand(0));
      break;

  case AMDGPU::BRANCH_COND_f32: {
    MachineInstr *NewMI =
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::PRED_X),
              AMDGPU::PREDICATE_BIT)
              .addOperand(MI->getOperand(1))
              .addImm(OPCODE_IS_NOT_ZERO)
              .addImm(0); // Flags
    TII->addFlag(NewMI, 0, MO_FLAG_PUSH);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP_COND))
            .addOperand(MI->getOperand(0))
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    break;
  }

  case AMDGPU::BRANCH_COND_i32: {
    MachineInstr *NewMI =
      BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::PRED_X),
            AMDGPU::PREDICATE_BIT)
            .addOperand(MI->getOperand(1))
            .addImm(OPCODE_IS_NOT_ZERO_INT)
            .addImm(0); // Flags
    TII->addFlag(NewMI, 0, MO_FLAG_PUSH);
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(AMDGPU::JUMP_COND))
           .addOperand(MI->getOperand(0))
            .addReg(AMDGPU::PREDICATE_BIT, RegState::Kill);
    break;
  }

  case AMDGPU::EG_ExportSwz:
  case AMDGPU::R600_ExportSwz: {
    // Instruction is left unmodified if its not the last one of its type
    bool isLastInstructionOfItsType = true;
    unsigned InstExportType = MI->getOperand(1).getImm();
    for (MachineBasicBlock::iterator NextExportInst = std::next(I),
         EndBlock = BB->end(); NextExportInst != EndBlock;
         NextExportInst = std::next(NextExportInst)) {
      if (NextExportInst->getOpcode() == AMDGPU::EG_ExportSwz ||
          NextExportInst->getOpcode() == AMDGPU::R600_ExportSwz) {
        unsigned CurrentInstExportType = NextExportInst->getOperand(1)
            .getImm();
        if (CurrentInstExportType == InstExportType) {
          isLastInstructionOfItsType = false;
          break;
        }
      }
    }
    bool EOP = isEOP(I);
    if (!EOP && !isLastInstructionOfItsType)
      return BB;
    unsigned CfInst = (MI->getOpcode() == AMDGPU::EG_ExportSwz)? 84 : 40;
    BuildMI(*BB, I, BB->findDebugLoc(I), TII->get(MI->getOpcode()))
            .addOperand(MI->getOperand(0))
            .addOperand(MI->getOperand(1))
            .addOperand(MI->getOperand(2))
            .addOperand(MI->getOperand(3))
            .addOperand(MI->getOperand(4))
            .addOperand(MI->getOperand(5))
            .addOperand(MI->getOperand(6))
            .addImm(CfInst)
            .addImm(EOP);
    break;
  }
  case AMDGPU::RETURN: {
    // RETURN instructions must have the live-out registers as implicit uses,
    // otherwise they appear dead.
    R600MachineFunctionInfo *MFI = MF->getInfo<R600MachineFunctionInfo>();
    MachineInstrBuilder MIB(*MF, MI);
    for (unsigned i = 0, e = MFI->LiveOuts.size(); i != e; ++i)
      MIB.addReg(MFI->LiveOuts[i], RegState::Implicit);
    return BB;
  }
  }

  MI->eraseFromParent();
  return BB;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

SDValue R600TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT: return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::SHL_PARTS: return LowerSHLParts(Op, DAG);
  case ISD::SRA_PARTS:
  case ISD::SRL_PARTS: return LowerSRXParts(Op, DAG);
  case ISD::UADDO: return LowerUADDSUBO(Op, DAG, ISD::ADD, AMDGPUISD::CARRY);
  case ISD::USUBO: return LowerUADDSUBO(Op, DAG, ISD::SUB, AMDGPUISD::BORROW);
  case ISD::FCOS:
  case ISD::FSIN: return LowerTrig(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::LOAD: {
    SDValue Result = LowerLOAD(Op, DAG);
    assert((!Result.getNode() ||
            Result.getNode()->getNumValues() == 2) &&
           "Load should return a value and a chain");
    return Result;
  }

  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::GlobalAddress: return LowerGlobalAddress(MFI, Op, DAG);
  case ISD::INTRINSIC_VOID: {
    SDValue Chain = Op.getOperand(0);
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    switch (IntrinsicID) {
    case AMDGPUIntrinsic::R600_store_swizzle: {
      SDLoc DL(Op);
      const SDValue Args[8] = {
        Chain,
        Op.getOperand(2), // Export Value
        Op.getOperand(3), // ArrayBase
        Op.getOperand(4), // Type
        DAG.getConstant(0, DL, MVT::i32), // SWZ_X
        DAG.getConstant(1, DL, MVT::i32), // SWZ_Y
        DAG.getConstant(2, DL, MVT::i32), // SWZ_Z
        DAG.getConstant(3, DL, MVT::i32) // SWZ_W
      };
      return DAG.getNode(AMDGPUISD::EXPORT, DL, Op.getValueType(), Args);
    }

    // default for switch(IntrinsicID)
    default: break;
    }
    // break out of case ISD::INTRINSIC_VOID in switch(Op.getOpcode())
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntrinsicID =
                         cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    EVT VT = Op.getValueType();
    SDLoc DL(Op);
    switch(IntrinsicID) {
    default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
    case AMDGPUIntrinsic::R600_interp_xy:
    case AMDGPUIntrinsic::R600_interp_zw: {
      int slot = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
      MachineSDNode *interp;
      SDValue RegisterINode = Op.getOperand(2);
      SDValue RegisterJNode = Op.getOperand(3);

      if (IntrinsicID == AMDGPUIntrinsic::R600_interp_xy)
        interp = DAG.getMachineNode(AMDGPU::INTERP_PAIR_XY, DL,
            MVT::f32, MVT::f32, DAG.getTargetConstant(slot, DL, MVT::i32),
            RegisterJNode, RegisterINode);
      else
        interp = DAG.getMachineNode(AMDGPU::INTERP_PAIR_ZW, DL,
            MVT::f32, MVT::f32, DAG.getTargetConstant(slot, DL, MVT::i32),
            RegisterJNode, RegisterINode);
      return DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2f32,
          SDValue(interp, 0), SDValue(interp, 1));
    }
    case AMDGPUIntrinsic::r600_tex:
    case AMDGPUIntrinsic::r600_texc:
    case AMDGPUIntrinsic::r600_txl:
    case AMDGPUIntrinsic::r600_txlc:
    case AMDGPUIntrinsic::r600_txb:
    case AMDGPUIntrinsic::r600_txbc:
    case AMDGPUIntrinsic::r600_txf:
    case AMDGPUIntrinsic::r600_txq:
    case AMDGPUIntrinsic::r600_ddx:
    case AMDGPUIntrinsic::r600_ddy:
    case AMDGPUIntrinsic::R600_ldptr: {
      unsigned TextureOp;
      switch (IntrinsicID) {
      case AMDGPUIntrinsic::r600_tex:
        TextureOp = 0;
        break;
      case AMDGPUIntrinsic::r600_texc:
        TextureOp = 1;
        break;
      case AMDGPUIntrinsic::r600_txl:
        TextureOp = 2;
        break;
      case AMDGPUIntrinsic::r600_txlc:
        TextureOp = 3;
        break;
      case AMDGPUIntrinsic::r600_txb:
        TextureOp = 4;
        break;
      case AMDGPUIntrinsic::r600_txbc:
        TextureOp = 5;
        break;
      case AMDGPUIntrinsic::r600_txf:
        TextureOp = 6;
        break;
      case AMDGPUIntrinsic::r600_txq:
        TextureOp = 7;
        break;
      case AMDGPUIntrinsic::r600_ddx:
        TextureOp = 8;
        break;
      case AMDGPUIntrinsic::r600_ddy:
        TextureOp = 9;
        break;
      case AMDGPUIntrinsic::R600_ldptr:
        TextureOp = 10;
        break;
      default:
        llvm_unreachable("Unknow Texture Operation");
      }

      SDValue TexArgs[19] = {
        DAG.getConstant(TextureOp, DL, MVT::i32),
        Op.getOperand(1),
        DAG.getConstant(0, DL, MVT::i32),
        DAG.getConstant(1, DL, MVT::i32),
        DAG.getConstant(2, DL, MVT::i32),
        DAG.getConstant(3, DL, MVT::i32),
        Op.getOperand(2),
        Op.getOperand(3),
        Op.getOperand(4),
        DAG.getConstant(0, DL, MVT::i32),
        DAG.getConstant(1, DL, MVT::i32),
        DAG.getConstant(2, DL, MVT::i32),
        DAG.getConstant(3, DL, MVT::i32),
        Op.getOperand(5),
        Op.getOperand(6),
        Op.getOperand(7),
        Op.getOperand(8),
        Op.getOperand(9),
        Op.getOperand(10)
      };
      return DAG.getNode(AMDGPUISD::TEXTURE_FETCH, DL, MVT::v4f32, TexArgs);
    }
    case AMDGPUIntrinsic::AMDGPU_dp4: {
      SDValue Args[8] = {
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(0, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(0, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(1, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(1, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(2, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(2, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(1),
          DAG.getConstant(3, DL, MVT::i32)),
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Op.getOperand(2),
          DAG.getConstant(3, DL, MVT::i32))
      };
      return DAG.getNode(AMDGPUISD::DOT4, DL, MVT::f32, Args);
    }

    case Intrinsic::r600_read_ngroups_x:
      return LowerImplicitParameter(DAG, VT, DL, 0);
    case Intrinsic::r600_read_ngroups_y:
      return LowerImplicitParameter(DAG, VT, DL, 1);
    case Intrinsic::r600_read_ngroups_z:
      return LowerImplicitParameter(DAG, VT, DL, 2);
    case Intrinsic::r600_read_global_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 3);
    case Intrinsic::r600_read_global_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 4);
    case Intrinsic::r600_read_global_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 5);
    case Intrinsic::r600_read_local_size_x:
      return LowerImplicitParameter(DAG, VT, DL, 6);
    case Intrinsic::r600_read_local_size_y:
      return LowerImplicitParameter(DAG, VT, DL, 7);
    case Intrinsic::r600_read_local_size_z:
      return LowerImplicitParameter(DAG, VT, DL, 8);

    case Intrinsic::r600_read_workdim:
    case AMDGPUIntrinsic::AMDGPU_read_workdim: { // Legacy name.
      uint32_t ByteOffset = getImplicitParameterOffset(MFI, GRID_DIM);
      return LowerImplicitParameter(DAG, VT, DL, ByteOffset / 4);
    }

    case Intrinsic::r600_read_tgid_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_X, VT);
    case Intrinsic::r600_read_tgid_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Y, VT);
    case Intrinsic::r600_read_tgid_z:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T1_Z, VT);
    case Intrinsic::r600_read_tidig_x:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_X, VT);
    case Intrinsic::r600_read_tidig_y:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_Y, VT);
    case Intrinsic::r600_read_tidig_z:
      return CreateLiveInRegister(DAG, &AMDGPU::R600_TReg32RegClass,
                                  AMDGPU::T0_Z, VT);

    // FIXME: Should be renamed to r600 prefix
    case Intrinsic::AMDGPU_rsq_clamped:
      return DAG.getNode(AMDGPUISD::RSQ_CLAMPED, DL, VT, Op.getOperand(1));

    case Intrinsic::r600_rsq:
      // XXX - I'm assuming SI's RSQ_LEGACY matches R600's behavior.
      return DAG.getNode(AMDGPUISD::RSQ_LEGACY, DL, VT, Op.getOperand(1));
    }
    // break out of case ISD::INTRINSIC_WO_CHAIN in switch(Op.getOpcode())
    break;
  }
  } // end switch(Op.getOpcode())
  return SDValue();
}

void R600TargetLowering::ReplaceNodeResults(SDNode *N,
                                            SmallVectorImpl<SDValue> &Results,
                                            SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    AMDGPUTargetLowering::ReplaceNodeResults(N, Results, DAG);
    return;
  case ISD::FP_TO_UINT:
    if (N->getValueType(0) == MVT::i1) {
      Results.push_back(LowerFPTOUINT(N->getOperand(0), DAG));
      return;
    }
    // Fall-through. Since we don't care about out of bounds values
    // we can use FP_TO_SINT for uints too. The DAGLegalizer code for uint
    // considers some extra cases which are not necessary here.
  case ISD::FP_TO_SINT: {
    SDValue Result;
    if (expandFP_TO_SINT(N, Result, DAG))
      Results.push_back(Result);
    return;
  }
  case ISD::SDIVREM: {
    SDValue Op = SDValue(N, 1);
    SDValue RES = LowerSDIVREM(Op, DAG);
    Results.push_back(RES);
    Results.push_back(RES.getValue(1));
    break;
  }
  case ISD::UDIVREM: {
    SDValue Op = SDValue(N, 0);
    LowerUDIVREM64(Op, DAG, Results);
    break;
  }
  }
}

SDValue R600TargetLowering::vectorToVerticalVector(SelectionDAG &DAG,
                                                   SDValue Vector) const {

  SDLoc DL(Vector);
  EVT VecVT = Vector.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  SmallVector<SDValue, 8> Args;

  for (unsigned i = 0, e = VecVT.getVectorNumElements();
                                                           i != e; ++i) {
    Args.push_back(DAG.getNode(
        ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Vector,
        DAG.getConstant(i, DL, getVectorIdxTy(DAG.getDataLayout()))));
  }

  return DAG.getNode(AMDGPUISD::BUILD_VERTICAL_VECTOR, DL, VecVT, Args);
}

SDValue R600TargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                    SelectionDAG &DAG) const {

  SDLoc DL(Op);
  SDValue Vector = Op.getOperand(0);
  SDValue Index = Op.getOperand(1);

  if (isa<ConstantSDNode>(Index) ||
      Vector.getOpcode() == AMDGPUISD::BUILD_VERTICAL_VECTOR)
    return Op;

  Vector = vectorToVerticalVector(DAG, Vector);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, Op.getValueType(),
                     Vector, Index);
}

SDValue R600TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Vector = Op.getOperand(0);
  SDValue Value = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);

  if (isa<ConstantSDNode>(Index) ||
      Vector.getOpcode() == AMDGPUISD::BUILD_VERTICAL_VECTOR)
    return Op;

  Vector = vectorToVerticalVector(DAG, Vector);
  SDValue Insert = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, Op.getValueType(),
                               Vector, Value, Index);
  return vectorToVerticalVector(DAG, Insert);
}

SDValue R600TargetLowering::LowerTrig(SDValue Op, SelectionDAG &DAG) const {
  // On hw >= R700, COS/SIN input must be between -1. and 1.
  // Thus we lower them to TRIG ( FRACT ( x / 2Pi + 0.5) - 0.5)
  EVT VT = Op.getValueType();
  SDValue Arg = Op.getOperand(0);
  SDLoc DL(Op);

  // TODO: Should this propagate fast-math-flags?
  SDValue FractPart = DAG.getNode(AMDGPUISD::FRACT, DL, VT,
      DAG.getNode(ISD::FADD, DL, VT,
        DAG.getNode(ISD::FMUL, DL, VT, Arg,
          DAG.getConstantFP(0.15915494309, DL, MVT::f32)),
        DAG.getConstantFP(0.5, DL, MVT::f32)));
  unsigned TrigNode;
  switch (Op.getOpcode()) {
  case ISD::FCOS:
    TrigNode = AMDGPUISD::COS_HW;
    break;
  case ISD::FSIN:
    TrigNode = AMDGPUISD::SIN_HW;
    break;
  default:
    llvm_unreachable("Wrong trig opcode");
  }
  SDValue TrigVal = DAG.getNode(TrigNode, DL, VT,
      DAG.getNode(ISD::FADD, DL, VT, FractPart,
        DAG.getConstantFP(-0.5, DL, MVT::f32)));
  if (Gen >= AMDGPUSubtarget::R700)
    return TrigVal;
  // On R600 hw, COS/SIN input must be between -Pi and Pi.
  return DAG.getNode(ISD::FMUL, DL, VT, TrigVal,
      DAG.getConstantFP(3.14159265359, DL, MVT::f32));
}

SDValue R600TargetLowering::LowerSHLParts(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shift = Op.getOperand(2);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One  = DAG.getConstant(1, DL, VT);

  SDValue Width  = DAG.getConstant(VT.getSizeInBits(), DL, VT);
  SDValue Width1 = DAG.getConstant(VT.getSizeInBits() - 1, DL, VT);
  SDValue BigShift  = DAG.getNode(ISD::SUB, DL, VT, Shift, Width);
  SDValue CompShift = DAG.getNode(ISD::SUB, DL, VT, Width1, Shift);

  // The dance around Width1 is necessary for 0 special case.
  // Without it the CompShift might be 32, producing incorrect results in
  // Overflow. So we do the shift in two steps, the alternative is to
  // add a conditional to filter the special case.

  SDValue Overflow = DAG.getNode(ISD::SRL, DL, VT, Lo, CompShift);
  Overflow = DAG.getNode(ISD::SRL, DL, VT, Overflow, One);

  SDValue HiSmall = DAG.getNode(ISD::SHL, DL, VT, Hi, Shift);
  HiSmall = DAG.getNode(ISD::OR, DL, VT, HiSmall, Overflow);
  SDValue LoSmall = DAG.getNode(ISD::SHL, DL, VT, Lo, Shift);

  SDValue HiBig = DAG.getNode(ISD::SHL, DL, VT, Lo, BigShift);
  SDValue LoBig = Zero;

  Hi = DAG.getSelectCC(DL, Shift, Width, HiSmall, HiBig, ISD::SETULT);
  Lo = DAG.getSelectCC(DL, Shift, Width, LoSmall, LoBig, ISD::SETULT);

  return DAG.getNode(ISD::MERGE_VALUES, DL, DAG.getVTList(VT,VT), Lo, Hi);
}

SDValue R600TargetLowering::LowerSRXParts(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shift = Op.getOperand(2);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One  = DAG.getConstant(1, DL, VT);

  const bool SRA = Op.getOpcode() == ISD::SRA_PARTS;

  SDValue Width  = DAG.getConstant(VT.getSizeInBits(), DL, VT);
  SDValue Width1 = DAG.getConstant(VT.getSizeInBits() - 1, DL, VT);
  SDValue BigShift  = DAG.getNode(ISD::SUB, DL, VT, Shift, Width);
  SDValue CompShift = DAG.getNode(ISD::SUB, DL, VT, Width1, Shift);

  // The dance around Width1 is necessary for 0 special case.
  // Without it the CompShift might be 32, producing incorrect results in
  // Overflow. So we do the shift in two steps, the alternative is to
  // add a conditional to filter the special case.

  SDValue Overflow = DAG.getNode(ISD::SHL, DL, VT, Hi, CompShift);
  Overflow = DAG.getNode(ISD::SHL, DL, VT, Overflow, One);

  SDValue HiSmall = DAG.getNode(SRA ? ISD::SRA : ISD::SRL, DL, VT, Hi, Shift);
  SDValue LoSmall = DAG.getNode(ISD::SRL, DL, VT, Lo, Shift);
  LoSmall = DAG.getNode(ISD::OR, DL, VT, LoSmall, Overflow);

  SDValue LoBig = DAG.getNode(SRA ? ISD::SRA : ISD::SRL, DL, VT, Hi, BigShift);
  SDValue HiBig = SRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, Width1) : Zero;

  Hi = DAG.getSelectCC(DL, Shift, Width, HiSmall, HiBig, ISD::SETULT);
  Lo = DAG.getSelectCC(DL, Shift, Width, LoSmall, LoBig, ISD::SETULT);

  return DAG.getNode(ISD::MERGE_VALUES, DL, DAG.getVTList(VT,VT), Lo, Hi);
}

SDValue R600TargetLowering::LowerUADDSUBO(SDValue Op, SelectionDAG &DAG,
                                          unsigned mainop, unsigned ovf) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);

  SDValue OVF = DAG.getNode(ovf, DL, VT, Lo, Hi);
  // Extend sign.
  OVF = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, OVF,
                    DAG.getValueType(MVT::i1));

  SDValue Res = DAG.getNode(mainop, DL, VT, Lo, Hi);

  return DAG.getNode(ISD::MERGE_VALUES, DL, DAG.getVTList(VT, VT), Res, OVF);
}

SDValue R600TargetLowering::LowerFPTOUINT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  return DAG.getNode(
      ISD::SETCC,
      DL,
      MVT::i1,
      Op, DAG.getConstantFP(0.0f, DL, MVT::f32),
      DAG.getCondCode(ISD::SETNE)
      );
}

SDValue R600TargetLowering::LowerImplicitParameter(SelectionDAG &DAG, EVT VT,
                                                   SDLoc DL,
                                                   unsigned DwordOffset) const {
  unsigned ByteOffset = DwordOffset * 4;
  PointerType * PtrType = PointerType::get(VT.getTypeForEVT(*DAG.getContext()),
                                      AMDGPUAS::CONSTANT_BUFFER_0);

  // We shouldn't be using an offset wider than 16-bits for implicit parameters.
  assert(isInt<16>(ByteOffset));

  return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                     DAG.getConstant(ByteOffset, DL, MVT::i32), // PTR
                     MachinePointerInfo(ConstantPointerNull::get(PtrType)),
                     false, false, false, 0);
}

bool R600TargetLowering::isZero(SDValue Op) const {
  if(ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(Op)) {
    return Cst->isNullValue();
  } else if(ConstantFPSDNode *CstFP = dyn_cast<ConstantFPSDNode>(Op)){
    return CstFP->isZero();
  } else {
    return false;
  }
}

SDValue R600TargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);
  SDValue Temp;

  if (VT == MVT::f32) {
    DAGCombinerInfo DCI(DAG, AfterLegalizeVectorOps, true, nullptr);
    SDValue MinMax = CombineFMinMaxLegacy(DL, VT, LHS, RHS, True, False, CC, DCI);
    if (MinMax)
      return MinMax;
  }

  // LHS and RHS are guaranteed to be the same value type
  EVT CompareVT = LHS.getValueType();

  // Check if we can lower this to a native operation.

  // Try to lower to a SET* instruction:
  //
  // SET* can match the following patterns:
  //
  // select_cc f32, f32, -1,  0, cc_supported
  // select_cc f32, f32, 1.0f, 0.0f, cc_supported
  // select_cc i32, i32, -1,  0, cc_supported
  //

  // Move hardware True/False values to the correct operand.
  ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
  ISD::CondCode InverseCC =
     ISD::getSetCCInverse(CCOpcode, CompareVT == MVT::i32);
  if (isHWTrueValue(False) && isHWFalseValue(True)) {
    if (isCondCodeLegal(InverseCC, CompareVT.getSimpleVT())) {
      std::swap(False, True);
      CC = DAG.getCondCode(InverseCC);
    } else {
      ISD::CondCode SwapInvCC = ISD::getSetCCSwappedOperands(InverseCC);
      if (isCondCodeLegal(SwapInvCC, CompareVT.getSimpleVT())) {
        std::swap(False, True);
        std::swap(LHS, RHS);
        CC = DAG.getCondCode(SwapInvCC);
      }
    }
  }

  if (isHWTrueValue(True) && isHWFalseValue(False) &&
      (CompareVT == VT || VT == MVT::i32)) {
    // This can be matched by a SET* instruction.
    return DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, True, False, CC);
  }

  // Try to lower to a CND* instruction:
  //
  // CND* can match the following patterns:
  //
  // select_cc f32, 0.0, f32, f32, cc_supported
  // select_cc f32, 0.0, i32, i32, cc_supported
  // select_cc i32, 0,   f32, f32, cc_supported
  // select_cc i32, 0,   i32, i32, cc_supported
  //

  // Try to move the zero value to the RHS
  if (isZero(LHS)) {
    ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
    // Try swapping the operands
    ISD::CondCode CCSwapped = ISD::getSetCCSwappedOperands(CCOpcode);
    if (isCondCodeLegal(CCSwapped, CompareVT.getSimpleVT())) {
      std::swap(LHS, RHS);
      CC = DAG.getCondCode(CCSwapped);
    } else {
      // Try inverting the conditon and then swapping the operands
      ISD::CondCode CCInv = ISD::getSetCCInverse(CCOpcode, CompareVT.isInteger());
      CCSwapped = ISD::getSetCCSwappedOperands(CCInv);
      if (isCondCodeLegal(CCSwapped, CompareVT.getSimpleVT())) {
        std::swap(True, False);
        std::swap(LHS, RHS);
        CC = DAG.getCondCode(CCSwapped);
      }
    }
  }
  if (isZero(RHS)) {
    SDValue Cond = LHS;
    SDValue Zero = RHS;
    ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
    if (CompareVT != VT) {
      // Bitcast True / False to the correct types.  This will end up being
      // a nop, but it allows us to define only a single pattern in the
      // .TD files for each CND* instruction rather than having to have
      // one pattern for integer True/False and one for fp True/False
      True = DAG.getNode(ISD::BITCAST, DL, CompareVT, True);
      False = DAG.getNode(ISD::BITCAST, DL, CompareVT, False);
    }

    switch (CCOpcode) {
    case ISD::SETONE:
    case ISD::SETUNE:
    case ISD::SETNE:
      CCOpcode = ISD::getSetCCInverse(CCOpcode, CompareVT == MVT::i32);
      Temp = True;
      True = False;
      False = Temp;
      break;
    default:
      break;
    }
    SDValue SelectNode = DAG.getNode(ISD::SELECT_CC, DL, CompareVT,
        Cond, Zero,
        True, False,
        DAG.getCondCode(CCOpcode));
    return DAG.getNode(ISD::BITCAST, DL, VT, SelectNode);
  }

  // If we make it this for it means we have no native instructions to handle
  // this SELECT_CC, so we must lower it.
  SDValue HWTrue, HWFalse;

  if (CompareVT == MVT::f32) {
    HWTrue = DAG.getConstantFP(1.0f, DL, CompareVT);
    HWFalse = DAG.getConstantFP(0.0f, DL, CompareVT);
  } else if (CompareVT == MVT::i32) {
    HWTrue = DAG.getConstant(-1, DL, CompareVT);
    HWFalse = DAG.getConstant(0, DL, CompareVT);
  }
  else {
    llvm_unreachable("Unhandled value type in LowerSELECT_CC");
  }

  // Lower this unsupported SELECT_CC into a combination of two supported
  // SELECT_CC operations.
  SDValue Cond = DAG.getNode(ISD::SELECT_CC, DL, CompareVT, LHS, RHS, HWTrue, HWFalse, CC);

  return DAG.getNode(ISD::SELECT_CC, DL, VT,
      Cond, HWFalse,
      True, False,
      DAG.getCondCode(ISD::SETNE));
}

/// LLVM generates byte-addressed pointers.  For indirect addressing, we need to
/// convert these pointers to a register index.  Each register holds
/// 16 bytes, (4 x 32bit sub-register), but we need to take into account the
/// \p StackWidth, which tells us how many of the 4 sub-registrers will be used
/// for indirect addressing.
SDValue R600TargetLowering::stackPtrToRegIndex(SDValue Ptr,
                                               unsigned StackWidth,
                                               SelectionDAG &DAG) const {
  unsigned SRLPad;
  switch(StackWidth) {
  case 1:
    SRLPad = 2;
    break;
  case 2:
    SRLPad = 3;
    break;
  case 4:
    SRLPad = 4;
    break;
  default: llvm_unreachable("Invalid stack width");
  }

  SDLoc DL(Ptr);
  return DAG.getNode(ISD::SRL, DL, Ptr.getValueType(), Ptr,
                     DAG.getConstant(SRLPad, DL, MVT::i32));
}

void R600TargetLowering::getStackAddress(unsigned StackWidth,
                                         unsigned ElemIdx,
                                         unsigned &Channel,
                                         unsigned &PtrIncr) const {
  switch (StackWidth) {
  default:
  case 1:
    Channel = 0;
    if (ElemIdx > 0) {
      PtrIncr = 1;
    } else {
      PtrIncr = 0;
    }
    break;
  case 2:
    Channel = ElemIdx % 2;
    if (ElemIdx == 2) {
      PtrIncr = 1;
    } else {
      PtrIncr = 0;
    }
    break;
  case 4:
    Channel = ElemIdx;
    PtrIncr = 0;
    break;
  }
}

SDValue R600TargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  StoreSDNode *StoreNode = cast<StoreSDNode>(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Value = Op.getOperand(1);
  SDValue Ptr = Op.getOperand(2);

  SDValue Result = AMDGPUTargetLowering::LowerSTORE(Op, DAG);
  if (Result.getNode()) {
    return Result;
  }

  if (StoreNode->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS) {
    if (StoreNode->isTruncatingStore()) {
      EVT VT = Value.getValueType();
      assert(VT.bitsLE(MVT::i32));
      EVT MemVT = StoreNode->getMemoryVT();
      SDValue MaskConstant;
      if (MemVT == MVT::i8) {
        MaskConstant = DAG.getConstant(0xFF, DL, MVT::i32);
      } else {
        assert(MemVT == MVT::i16);
        MaskConstant = DAG.getConstant(0xFFFF, DL, MVT::i32);
      }
      SDValue DWordAddr = DAG.getNode(ISD::SRL, DL, VT, Ptr,
                                      DAG.getConstant(2, DL, MVT::i32));
      SDValue ByteIndex = DAG.getNode(ISD::AND, DL, Ptr.getValueType(), Ptr,
                                      DAG.getConstant(0x00000003, DL, VT));
      SDValue TruncValue = DAG.getNode(ISD::AND, DL, VT, Value, MaskConstant);
      SDValue Shift = DAG.getNode(ISD::SHL, DL, VT, ByteIndex,
                                   DAG.getConstant(3, DL, VT));
      SDValue ShiftedValue = DAG.getNode(ISD::SHL, DL, VT, TruncValue, Shift);
      SDValue Mask = DAG.getNode(ISD::SHL, DL, VT, MaskConstant, Shift);
      // XXX: If we add a 64-bit ZW register class, then we could use a 2 x i32
      // vector instead.
      SDValue Src[4] = {
        ShiftedValue,
        DAG.getConstant(0, DL, MVT::i32),
        DAG.getConstant(0, DL, MVT::i32),
        Mask
      };
      SDValue Input = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v4i32, Src);
      SDValue Args[3] = { Chain, Input, DWordAddr };
      return DAG.getMemIntrinsicNode(AMDGPUISD::STORE_MSKOR, DL,
                                     Op->getVTList(), Args, MemVT,
                                     StoreNode->getMemOperand());
    } else if (Ptr->getOpcode() != AMDGPUISD::DWORDADDR &&
               Value.getValueType().bitsGE(MVT::i32)) {
      // Convert pointer from byte address to dword address.
      Ptr = DAG.getNode(AMDGPUISD::DWORDADDR, DL, Ptr.getValueType(),
                        DAG.getNode(ISD::SRL, DL, Ptr.getValueType(),
                                    Ptr, DAG.getConstant(2, DL, MVT::i32)));

      if (StoreNode->isTruncatingStore() || StoreNode->isIndexed()) {
        llvm_unreachable("Truncated and indexed stores not supported yet");
      } else {
        Chain = DAG.getStore(Chain, DL, Value, Ptr, StoreNode->getMemOperand());
      }
      return Chain;
    }
  }

  EVT ValueVT = Value.getValueType();

  if (StoreNode->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS) {
    return SDValue();
  }

  SDValue Ret = AMDGPUTargetLowering::LowerSTORE(Op, DAG);
  if (Ret.getNode()) {
    return Ret;
  }
  // Lowering for indirect addressing

  const MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL =
      static_cast<const AMDGPUFrameLowering *>(Subtarget->getFrameLowering());
  unsigned StackWidth = TFL->getStackWidth(MF);

  Ptr = stackPtrToRegIndex(Ptr, StackWidth, DAG);

  if (ValueVT.isVector()) {
    unsigned NumElemVT = ValueVT.getVectorNumElements();
    EVT ElemVT = ValueVT.getVectorElementType();
    SmallVector<SDValue, 4> Stores(NumElemVT);

    assert(NumElemVT >= StackWidth && "Stack width cannot be greater than "
                                      "vector width in load");

    for (unsigned i = 0; i < NumElemVT; ++i) {
      unsigned Channel, PtrIncr;
      getStackAddress(StackWidth, i, Channel, PtrIncr);
      Ptr = DAG.getNode(ISD::ADD, DL, MVT::i32, Ptr,
                        DAG.getConstant(PtrIncr, DL, MVT::i32));
      SDValue Elem = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ElemVT,
                                 Value, DAG.getConstant(i, DL, MVT::i32));

      Stores[i] = DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other,
                              Chain, Elem, Ptr,
                              DAG.getTargetConstant(Channel, DL, MVT::i32));
    }
     Chain =  DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Stores);
   } else {
    if (ValueVT == MVT::i8) {
      Value = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Value);
    }
    Chain = DAG.getNode(AMDGPUISD::REGISTER_STORE, DL, MVT::Other, Chain, Value, Ptr,
    DAG.getTargetConstant(0, DL, MVT::i32)); // Channel
  }

  return Chain;
}

// return (512 + (kc_bank << 12)
static int
ConstantAddressBlock(unsigned AddressSpace) {
  switch (AddressSpace) {
  case AMDGPUAS::CONSTANT_BUFFER_0:
    return 512;
  case AMDGPUAS::CONSTANT_BUFFER_1:
    return 512 + 4096;
  case AMDGPUAS::CONSTANT_BUFFER_2:
    return 512 + 4096 * 2;
  case AMDGPUAS::CONSTANT_BUFFER_3:
    return 512 + 4096 * 3;
  case AMDGPUAS::CONSTANT_BUFFER_4:
    return 512 + 4096 * 4;
  case AMDGPUAS::CONSTANT_BUFFER_5:
    return 512 + 4096 * 5;
  case AMDGPUAS::CONSTANT_BUFFER_6:
    return 512 + 4096 * 6;
  case AMDGPUAS::CONSTANT_BUFFER_7:
    return 512 + 4096 * 7;
  case AMDGPUAS::CONSTANT_BUFFER_8:
    return 512 + 4096 * 8;
  case AMDGPUAS::CONSTANT_BUFFER_9:
    return 512 + 4096 * 9;
  case AMDGPUAS::CONSTANT_BUFFER_10:
    return 512 + 4096 * 10;
  case AMDGPUAS::CONSTANT_BUFFER_11:
    return 512 + 4096 * 11;
  case AMDGPUAS::CONSTANT_BUFFER_12:
    return 512 + 4096 * 12;
  case AMDGPUAS::CONSTANT_BUFFER_13:
    return 512 + 4096 * 13;
  case AMDGPUAS::CONSTANT_BUFFER_14:
    return 512 + 4096 * 14;
  case AMDGPUAS::CONSTANT_BUFFER_15:
    return 512 + 4096 * 15;
  default:
    return -1;
  }
}

SDValue R600TargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const
{
  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  LoadSDNode *LoadNode = cast<LoadSDNode>(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Ptr = Op.getOperand(1);
  SDValue LoweredLoad;

  if (SDValue Ret = AMDGPUTargetLowering::LowerLOAD(Op, DAG))
    return Ret;

  // Lower loads constant address space global variable loads
  if (LoadNode->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS &&
      isa<GlobalVariable>(GetUnderlyingObject(
          LoadNode->getMemOperand()->getValue(), DAG.getDataLayout()))) {

    SDValue Ptr = DAG.getZExtOrTrunc(
        LoadNode->getBasePtr(), DL,
        getPointerTy(DAG.getDataLayout(), AMDGPUAS::PRIVATE_ADDRESS));
    Ptr = DAG.getNode(ISD::SRL, DL, MVT::i32, Ptr,
        DAG.getConstant(2, DL, MVT::i32));
    return DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, Op->getVTList(),
                       LoadNode->getChain(), Ptr,
                       DAG.getTargetConstant(0, DL, MVT::i32),
                       Op.getOperand(2));
  }

  if (LoadNode->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS && VT.isVector()) {
    SDValue MergedValues[2] = {
      ScalarizeVectorLoad(Op, DAG),
      Chain
    };
    return DAG.getMergeValues(MergedValues, DL);
  }

  int ConstantBlock = ConstantAddressBlock(LoadNode->getAddressSpace());
  if (ConstantBlock > -1 &&
      ((LoadNode->getExtensionType() == ISD::NON_EXTLOAD) ||
       (LoadNode->getExtensionType() == ISD::ZEXTLOAD))) {
    SDValue Result;
    if (isa<ConstantExpr>(LoadNode->getMemOperand()->getValue()) ||
        isa<Constant>(LoadNode->getMemOperand()->getValue()) ||
        isa<ConstantSDNode>(Ptr)) {
      SDValue Slots[4];
      for (unsigned i = 0; i < 4; i++) {
        // We want Const position encoded with the following formula :
        // (((512 + (kc_bank << 12) + const_index) << 2) + chan)
        // const_index is Ptr computed by llvm using an alignment of 16.
        // Thus we add (((512 + (kc_bank << 12)) + chan ) * 4 here and
        // then div by 4 at the ISel step
        SDValue NewPtr = DAG.getNode(ISD::ADD, DL, Ptr.getValueType(), Ptr,
            DAG.getConstant(4 * i + ConstantBlock * 16, DL, MVT::i32));
        Slots[i] = DAG.getNode(AMDGPUISD::CONST_ADDRESS, DL, MVT::i32, NewPtr);
      }
      EVT NewVT = MVT::v4i32;
      unsigned NumElements = 4;
      if (VT.isVector()) {
        NewVT = VT;
        NumElements = VT.getVectorNumElements();
      }
      Result = DAG.getNode(ISD::BUILD_VECTOR, DL, NewVT,
                           makeArrayRef(Slots, NumElements));
    } else {
      // non-constant ptr can't be folded, keeps it as a v4f32 load
      Result = DAG.getNode(AMDGPUISD::CONST_ADDRESS, DL, MVT::v4i32,
          DAG.getNode(ISD::SRL, DL, MVT::i32, Ptr,
                      DAG.getConstant(4, DL, MVT::i32)),
                      DAG.getConstant(LoadNode->getAddressSpace() -
                                      AMDGPUAS::CONSTANT_BUFFER_0, DL, MVT::i32)
          );
    }

    if (!VT.isVector()) {
      Result = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, Result,
                           DAG.getConstant(0, DL, MVT::i32));
    }

    SDValue MergedValues[2] = {
      Result,
      Chain
    };
    return DAG.getMergeValues(MergedValues, DL);
  }

  // For most operations returning SDValue() will result in the node being
  // expanded by the DAG Legalizer. This is not the case for ISD::LOAD, so we
  // need to manually expand loads that may be legal in some address spaces and
  // illegal in others. SEXT loads from CONSTANT_BUFFER_0 are supported for
  // compute shaders, since the data is sign extended when it is uploaded to the
  // buffer. However SEXT loads from other address spaces are not supported, so
  // we need to expand them here.
  if (LoadNode->getExtensionType() == ISD::SEXTLOAD) {
    EVT MemVT = LoadNode->getMemoryVT();
    assert(!MemVT.isVector() && (MemVT == MVT::i16 || MemVT == MVT::i8));
    SDValue NewLoad = DAG.getExtLoad(ISD::EXTLOAD, DL, VT, Chain, Ptr,
                                  LoadNode->getPointerInfo(), MemVT,
                                  LoadNode->isVolatile(),
                                  LoadNode->isNonTemporal(),
                                  LoadNode->isInvariant(),
                                  LoadNode->getAlignment());
    SDValue Res = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, NewLoad,
                              DAG.getValueType(MemVT));

    SDValue MergedValues[2] = { Res, Chain };
    return DAG.getMergeValues(MergedValues, DL);
  }

  if (LoadNode->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS) {
    return SDValue();
  }

  // Lowering for indirect addressing
  const MachineFunction &MF = DAG.getMachineFunction();
  const AMDGPUFrameLowering *TFL =
      static_cast<const AMDGPUFrameLowering *>(Subtarget->getFrameLowering());
  unsigned StackWidth = TFL->getStackWidth(MF);

  Ptr = stackPtrToRegIndex(Ptr, StackWidth, DAG);

  if (VT.isVector()) {
    unsigned NumElemVT = VT.getVectorNumElements();
    EVT ElemVT = VT.getVectorElementType();
    SDValue Loads[4];

    assert(NumElemVT >= StackWidth && "Stack width cannot be greater than "
                                      "vector width in load");

    for (unsigned i = 0; i < NumElemVT; ++i) {
      unsigned Channel, PtrIncr;
      getStackAddress(StackWidth, i, Channel, PtrIncr);
      Ptr = DAG.getNode(ISD::ADD, DL, MVT::i32, Ptr,
                        DAG.getConstant(PtrIncr, DL, MVT::i32));
      Loads[i] = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, ElemVT,
                             Chain, Ptr,
                             DAG.getTargetConstant(Channel, DL, MVT::i32),
                             Op.getOperand(2));
    }
    for (unsigned i = NumElemVT; i < 4; ++i) {
      Loads[i] = DAG.getUNDEF(ElemVT);
    }
    EVT TargetVT = EVT::getVectorVT(*DAG.getContext(), ElemVT, 4);
    LoweredLoad = DAG.getNode(ISD::BUILD_VECTOR, DL, TargetVT, Loads);
  } else {
    LoweredLoad = DAG.getNode(AMDGPUISD::REGISTER_LOAD, DL, VT,
                              Chain, Ptr,
                              DAG.getTargetConstant(0, DL, MVT::i32), // Channel
                              Op.getOperand(2));
  }

  SDValue Ops[2] = {
    LoweredLoad,
    Chain
  };

  return DAG.getMergeValues(Ops, DL);
}

SDValue R600TargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Jump  = Op.getOperand(2);

  return DAG.getNode(AMDGPUISD::BRANCH_COND, SDLoc(Op), Op.getValueType(),
                     Chain, Jump, Cond);
}

/// XXX Only kernel functions are supported, so we can assume for now that
/// every function is a kernel function, but in the future we should use
/// separate calling conventions for kernel and non-kernel functions.
SDValue R600TargetLowering::LowerFormalArguments(
                                      SDValue Chain,
                                      CallingConv::ID CallConv,
                                      bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc DL, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());
  MachineFunction &MF = DAG.getMachineFunction();
  R600MachineFunctionInfo *MFI = MF.getInfo<R600MachineFunctionInfo>();

  SmallVector<ISD::InputArg, 8> LocalIns;

  getOriginalFunctionArgs(DAG, MF.getFunction(), Ins, LocalIns);

  AnalyzeFormalArguments(CCInfo, LocalIns);

  for (unsigned i = 0, e = Ins.size(); i < e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    const ISD::InputArg &In = Ins[i];
    EVT VT = In.VT;
    EVT MemVT = VA.getLocVT();
    if (!VT.isVector() && MemVT.isVector()) {
      // Get load source type if scalarized.
      MemVT = MemVT.getVectorElementType();
    }

    if (MFI->getShaderType() != ShaderType::COMPUTE) {
      unsigned Reg = MF.addLiveIn(VA.getLocReg(), &AMDGPU::R600_Reg128RegClass);
      SDValue Register = DAG.getCopyFromReg(Chain, DL, Reg, VT);
      InVals.push_back(Register);
      continue;
    }

    PointerType *PtrTy = PointerType::get(VT.getTypeForEVT(*DAG.getContext()),
                                          AMDGPUAS::CONSTANT_BUFFER_0);

    // i64 isn't a legal type, so the register type used ends up as i32, which
    // isn't expected here. It attempts to create this sextload, but it ends up
    // being invalid. Somehow this seems to work with i64 arguments, but breaks
    // for <1 x i64>.

    // The first 36 bytes of the input buffer contains information about
    // thread group and global sizes.
    ISD::LoadExtType Ext = ISD::NON_EXTLOAD;
    if (MemVT.getScalarSizeInBits() != VT.getScalarSizeInBits()) {
      // FIXME: This should really check the extload type, but the handling of
      // extload vector parameters seems to be broken.

      // Ext = In.Flags.isSExt() ? ISD::SEXTLOAD : ISD::ZEXTLOAD;
      Ext = ISD::SEXTLOAD;
    }

    // Compute the offset from the value.
    // XXX - I think PartOffset should give you this, but it seems to give the
    // size of the register which isn't useful.

    unsigned ValBase = ArgLocs[In.getOrigArgIndex()].getLocMemOffset();
    unsigned PartOffset = VA.getLocMemOffset();
    unsigned Offset = 36 + VA.getLocMemOffset();

    MachinePointerInfo PtrInfo(UndefValue::get(PtrTy), PartOffset - ValBase);
    SDValue Arg = DAG.getLoad(ISD::UNINDEXED, Ext, VT, DL, Chain,
                              DAG.getConstant(Offset, DL, MVT::i32),
                              DAG.getUNDEF(MVT::i32),
                              PtrInfo,
                              MemVT, false, true, true, 4);

    // 4 is the preferred alignment for the CONSTANT memory space.
    InVals.push_back(Arg);
    MFI->ABIArgOffset = Offset + MemVT.getStoreSize();
  }
  return Chain;
}

EVT R600TargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &,
                                           EVT VT) const {
   if (!VT.isVector())
     return MVT::i32;
   return VT.changeVectorElementTypeToInteger();
}

static SDValue CompactSwizzlableVector(
  SelectionDAG &DAG, SDValue VectorEntry,
  DenseMap<unsigned, unsigned> &RemapSwizzle) {
  assert(VectorEntry.getOpcode() == ISD::BUILD_VECTOR);
  assert(RemapSwizzle.empty());
  SDValue NewBldVec[4] = {
    VectorEntry.getOperand(0),
    VectorEntry.getOperand(1),
    VectorEntry.getOperand(2),
    VectorEntry.getOperand(3)
  };

  for (unsigned i = 0; i < 4; i++) {
    if (NewBldVec[i].getOpcode() == ISD::UNDEF)
      // We mask write here to teach later passes that the ith element of this
      // vector is undef. Thus we can use it to reduce 128 bits reg usage,
      // break false dependencies and additionnaly make assembly easier to read.
      RemapSwizzle[i] = 7; // SEL_MASK_WRITE
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(NewBldVec[i])) {
      if (C->isZero()) {
        RemapSwizzle[i] = 4; // SEL_0
        NewBldVec[i] = DAG.getUNDEF(MVT::f32);
      } else if (C->isExactlyValue(1.0)) {
        RemapSwizzle[i] = 5; // SEL_1
        NewBldVec[i] = DAG.getUNDEF(MVT::f32);
      }
    }

    if (NewBldVec[i].getOpcode() == ISD::UNDEF)
      continue;
    for (unsigned j = 0; j < i; j++) {
      if (NewBldVec[i] == NewBldVec[j]) {
        NewBldVec[i] = DAG.getUNDEF(NewBldVec[i].getValueType());
        RemapSwizzle[i] = j;
        break;
      }
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(VectorEntry),
                     VectorEntry.getValueType(), NewBldVec);
}

static SDValue ReorganizeVector(SelectionDAG &DAG, SDValue VectorEntry,
                                DenseMap<unsigned, unsigned> &RemapSwizzle) {
  assert(VectorEntry.getOpcode() == ISD::BUILD_VECTOR);
  assert(RemapSwizzle.empty());
  SDValue NewBldVec[4] = {
      VectorEntry.getOperand(0),
      VectorEntry.getOperand(1),
      VectorEntry.getOperand(2),
      VectorEntry.getOperand(3)
  };
  bool isUnmovable[4] = { false, false, false, false };
  for (unsigned i = 0; i < 4; i++) {
    RemapSwizzle[i] = i;
    if (NewBldVec[i].getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
      unsigned Idx = dyn_cast<ConstantSDNode>(NewBldVec[i].getOperand(1))
          ->getZExtValue();
      if (i == Idx)
        isUnmovable[Idx] = true;
    }
  }

  for (unsigned i = 0; i < 4; i++) {
    if (NewBldVec[i].getOpcode() == ISD::EXTRACT_VECTOR_ELT) {
      unsigned Idx = dyn_cast<ConstantSDNode>(NewBldVec[i].getOperand(1))
          ->getZExtValue();
      if (isUnmovable[Idx])
        continue;
      // Swap i and Idx
      std::swap(NewBldVec[Idx], NewBldVec[i]);
      std::swap(RemapSwizzle[i], RemapSwizzle[Idx]);
      break;
    }
  }

  return DAG.getNode(ISD::BUILD_VECTOR, SDLoc(VectorEntry),
                     VectorEntry.getValueType(), NewBldVec);
}


SDValue R600TargetLowering::OptimizeSwizzle(SDValue BuildVector,
                                            SDValue Swz[4], SelectionDAG &DAG,
                                            SDLoc DL) const {
  assert(BuildVector.getOpcode() == ISD::BUILD_VECTOR);
  // Old -> New swizzle values
  DenseMap<unsigned, unsigned> SwizzleRemap;

  BuildVector = CompactSwizzlableVector(DAG, BuildVector, SwizzleRemap);
  for (unsigned i = 0; i < 4; i++) {
    unsigned Idx = cast<ConstantSDNode>(Swz[i])->getZExtValue();
    if (SwizzleRemap.find(Idx) != SwizzleRemap.end())
      Swz[i] = DAG.getConstant(SwizzleRemap[Idx], DL, MVT::i32);
  }

  SwizzleRemap.clear();
  BuildVector = ReorganizeVector(DAG, BuildVector, SwizzleRemap);
  for (unsigned i = 0; i < 4; i++) {
    unsigned Idx = cast<ConstantSDNode>(Swz[i])->getZExtValue();
    if (SwizzleRemap.find(Idx) != SwizzleRemap.end())
      Swz[i] = DAG.getConstant(SwizzleRemap[Idx], DL, MVT::i32);
  }

  return BuildVector;
}


//===----------------------------------------------------------------------===//
// Custom DAG Optimizations
//===----------------------------------------------------------------------===//

SDValue R600TargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
  default: return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
  // (f32 fp_round (f64 uint_to_fp a)) -> (f32 uint_to_fp a)
  case ISD::FP_ROUND: {
      SDValue Arg = N->getOperand(0);
      if (Arg.getOpcode() == ISD::UINT_TO_FP && Arg.getValueType() == MVT::f64) {
        return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), N->getValueType(0),
                           Arg.getOperand(0));
      }
      break;
    }

  // (i32 fp_to_sint (fneg (select_cc f32, f32, 1.0, 0.0 cc))) ->
  // (i32 select_cc f32, f32, -1, 0 cc)
  //
  // Mesa's GLSL frontend generates the above pattern a lot and we can lower
  // this to one of the SET*_DX10 instructions.
  case ISD::FP_TO_SINT: {
    SDValue FNeg = N->getOperand(0);
    if (FNeg.getOpcode() != ISD::FNEG) {
      return SDValue();
    }
    SDValue SelectCC = FNeg.getOperand(0);
    if (SelectCC.getOpcode() != ISD::SELECT_CC ||
        SelectCC.getOperand(0).getValueType() != MVT::f32 || // LHS
        SelectCC.getOperand(2).getValueType() != MVT::f32 || // True
        !isHWTrueValue(SelectCC.getOperand(2)) ||
        !isHWFalseValue(SelectCC.getOperand(3))) {
      return SDValue();
    }

    SDLoc dl(N);
    return DAG.getNode(ISD::SELECT_CC, dl, N->getValueType(0),
                           SelectCC.getOperand(0), // LHS
                           SelectCC.getOperand(1), // RHS
                           DAG.getConstant(-1, dl, MVT::i32), // True
                           DAG.getConstant(0, dl, MVT::i32),  // False
                           SelectCC.getOperand(4)); // CC

    break;
  }

  // insert_vector_elt (build_vector elt0, ... , eltN), NewEltIdx, idx
  // => build_vector elt0, ... , NewEltIdx, ... , eltN
  case ISD::INSERT_VECTOR_ELT: {
    SDValue InVec = N->getOperand(0);
    SDValue InVal = N->getOperand(1);
    SDValue EltNo = N->getOperand(2);
    SDLoc dl(N);

    // If the inserted element is an UNDEF, just use the input vector.
    if (InVal.getOpcode() == ISD::UNDEF)
      return InVec;

    EVT VT = InVec.getValueType();

    // If we can't generate a legal BUILD_VECTOR, exit
    if (!isOperationLegal(ISD::BUILD_VECTOR, VT))
      return SDValue();

    // Check that we know which element is being inserted
    if (!isa<ConstantSDNode>(EltNo))
      return SDValue();
    unsigned Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();

    // Check that the operand is a BUILD_VECTOR (or UNDEF, which can essentially
    // be converted to a BUILD_VECTOR).  Fill in the Ops vector with the
    // vector elements.
    SmallVector<SDValue, 8> Ops;
    if (InVec.getOpcode() == ISD::BUILD_VECTOR) {
      Ops.append(InVec.getNode()->op_begin(),
                 InVec.getNode()->op_end());
    } else if (InVec.getOpcode() == ISD::UNDEF) {
      unsigned NElts = VT.getVectorNumElements();
      Ops.append(NElts, DAG.getUNDEF(InVal.getValueType()));
    } else {
      return SDValue();
    }

    // Insert the element
    if (Elt < Ops.size()) {
      // All the operands of BUILD_VECTOR must have the same type;
      // we enforce that here.
      EVT OpVT = Ops[0].getValueType();
      if (InVal.getValueType() != OpVT)
        InVal = OpVT.bitsGT(InVal.getValueType()) ?
          DAG.getNode(ISD::ANY_EXTEND, dl, OpVT, InVal) :
          DAG.getNode(ISD::TRUNCATE, dl, OpVT, InVal);
      Ops[Elt] = InVal;
    }

    // Return the new vector
    return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, Ops);
  }

  // Extract_vec (Build_vector) generated by custom lowering
  // also needs to be customly combined
  case ISD::EXTRACT_VECTOR_ELT: {
    SDValue Arg = N->getOperand(0);
    if (Arg.getOpcode() == ISD::BUILD_VECTOR) {
      if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
        unsigned Element = Const->getZExtValue();
        return Arg->getOperand(Element);
      }
    }
    if (Arg.getOpcode() == ISD::BITCAST &&
        Arg.getOperand(0).getOpcode() == ISD::BUILD_VECTOR) {
      if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
        unsigned Element = Const->getZExtValue();
        return DAG.getNode(ISD::BITCAST, SDLoc(N), N->getVTList(),
            Arg->getOperand(0).getOperand(Element));
      }
    }
    break;
  }

  case ISD::SELECT_CC: {
    // Try common optimizations
    SDValue Ret = AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
    if (Ret.getNode())
      return Ret;

    // fold selectcc (selectcc x, y, a, b, cc), b, a, b, seteq ->
    //      selectcc x, y, a, b, inv(cc)
    //
    // fold selectcc (selectcc x, y, a, b, cc), b, a, b, setne ->
    //      selectcc x, y, a, b, cc
    SDValue LHS = N->getOperand(0);
    if (LHS.getOpcode() != ISD::SELECT_CC) {
      return SDValue();
    }

    SDValue RHS = N->getOperand(1);
    SDValue True = N->getOperand(2);
    SDValue False = N->getOperand(3);
    ISD::CondCode NCC = cast<CondCodeSDNode>(N->getOperand(4))->get();

    if (LHS.getOperand(2).getNode() != True.getNode() ||
        LHS.getOperand(3).getNode() != False.getNode() ||
        RHS.getNode() != False.getNode()) {
      return SDValue();
    }

    switch (NCC) {
    default: return SDValue();
    case ISD::SETNE: return LHS;
    case ISD::SETEQ: {
      ISD::CondCode LHSCC = cast<CondCodeSDNode>(LHS.getOperand(4))->get();
      LHSCC = ISD::getSetCCInverse(LHSCC,
                                  LHS.getOperand(0).getValueType().isInteger());
      if (DCI.isBeforeLegalizeOps() ||
          isCondCodeLegal(LHSCC, LHS.getOperand(0).getSimpleValueType()))
        return DAG.getSelectCC(SDLoc(N),
                               LHS.getOperand(0),
                               LHS.getOperand(1),
                               LHS.getOperand(2),
                               LHS.getOperand(3),
                               LHSCC);
      break;
    }
    }
    return SDValue();
  }

  case AMDGPUISD::EXPORT: {
    SDValue Arg = N->getOperand(1);
    if (Arg.getOpcode() != ISD::BUILD_VECTOR)
      break;

    SDValue NewArgs[8] = {
      N->getOperand(0), // Chain
      SDValue(),
      N->getOperand(2), // ArrayBase
      N->getOperand(3), // Type
      N->getOperand(4), // SWZ_X
      N->getOperand(5), // SWZ_Y
      N->getOperand(6), // SWZ_Z
      N->getOperand(7) // SWZ_W
    };
    SDLoc DL(N);
    NewArgs[1] = OptimizeSwizzle(N->getOperand(1), &NewArgs[4], DAG, DL);
    return DAG.getNode(AMDGPUISD::EXPORT, DL, N->getVTList(), NewArgs);
  }
  case AMDGPUISD::TEXTURE_FETCH: {
    SDValue Arg = N->getOperand(1);
    if (Arg.getOpcode() != ISD::BUILD_VECTOR)
      break;

    SDValue NewArgs[19] = {
      N->getOperand(0),
      N->getOperand(1),
      N->getOperand(2),
      N->getOperand(3),
      N->getOperand(4),
      N->getOperand(5),
      N->getOperand(6),
      N->getOperand(7),
      N->getOperand(8),
      N->getOperand(9),
      N->getOperand(10),
      N->getOperand(11),
      N->getOperand(12),
      N->getOperand(13),
      N->getOperand(14),
      N->getOperand(15),
      N->getOperand(16),
      N->getOperand(17),
      N->getOperand(18),
    };
    SDLoc DL(N);
    NewArgs[1] = OptimizeSwizzle(N->getOperand(1), &NewArgs[2], DAG, DL);
    return DAG.getNode(AMDGPUISD::TEXTURE_FETCH, DL, N->getVTList(), NewArgs);
  }
  }

  return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
}

static bool
FoldOperand(SDNode *ParentNode, unsigned SrcIdx, SDValue &Src, SDValue &Neg,
            SDValue &Abs, SDValue &Sel, SDValue &Imm, SelectionDAG &DAG) {
  const R600InstrInfo *TII =
      static_cast<const R600InstrInfo *>(DAG.getSubtarget().getInstrInfo());
  if (!Src.isMachineOpcode())
    return false;
  switch (Src.getMachineOpcode()) {
  case AMDGPU::FNEG_R600:
    if (!Neg.getNode())
      return false;
    Src = Src.getOperand(0);
    Neg = DAG.getTargetConstant(1, SDLoc(ParentNode), MVT::i32);
    return true;
  case AMDGPU::FABS_R600:
    if (!Abs.getNode())
      return false;
    Src = Src.getOperand(0);
    Abs = DAG.getTargetConstant(1, SDLoc(ParentNode), MVT::i32);
    return true;
  case AMDGPU::CONST_COPY: {
    unsigned Opcode = ParentNode->getMachineOpcode();
    bool HasDst = TII->getOperandIdx(Opcode, AMDGPU::OpName::dst) > -1;

    if (!Sel.getNode())
      return false;

    SDValue CstOffset = Src.getOperand(0);
    if (ParentNode->getValueType(0).isVector())
      return false;

    // Gather constants values
    int SrcIndices[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src2),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_W),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_W)
    };
    std::vector<unsigned> Consts;
    for (int OtherSrcIdx : SrcIndices) {
      int OtherSelIdx = TII->getSelIdx(Opcode, OtherSrcIdx);
      if (OtherSrcIdx < 0 || OtherSelIdx < 0)
        continue;
      if (HasDst) {
        OtherSrcIdx--;
        OtherSelIdx--;
      }
      if (RegisterSDNode *Reg =
          dyn_cast<RegisterSDNode>(ParentNode->getOperand(OtherSrcIdx))) {
        if (Reg->getReg() == AMDGPU::ALU_CONST) {
          ConstantSDNode *Cst
            = cast<ConstantSDNode>(ParentNode->getOperand(OtherSelIdx));
          Consts.push_back(Cst->getZExtValue());
        }
      }
    }

    ConstantSDNode *Cst = cast<ConstantSDNode>(CstOffset);
    Consts.push_back(Cst->getZExtValue());
    if (!TII->fitsConstReadLimitations(Consts)) {
      return false;
    }

    Sel = CstOffset;
    Src = DAG.getRegister(AMDGPU::ALU_CONST, MVT::f32);
    return true;
  }
  case AMDGPU::MOV_IMM_I32:
  case AMDGPU::MOV_IMM_F32: {
    unsigned ImmReg = AMDGPU::ALU_LITERAL_X;
    uint64_t ImmValue = 0;


    if (Src.getMachineOpcode() == AMDGPU::MOV_IMM_F32) {
      ConstantFPSDNode *FPC = dyn_cast<ConstantFPSDNode>(Src.getOperand(0));
      float FloatValue = FPC->getValueAPF().convertToFloat();
      if (FloatValue == 0.0) {
        ImmReg = AMDGPU::ZERO;
      } else if (FloatValue == 0.5) {
        ImmReg = AMDGPU::HALF;
      } else if (FloatValue == 1.0) {
        ImmReg = AMDGPU::ONE;
      } else {
        ImmValue = FPC->getValueAPF().bitcastToAPInt().getZExtValue();
      }
    } else {
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(Src.getOperand(0));
      uint64_t Value = C->getZExtValue();
      if (Value == 0) {
        ImmReg = AMDGPU::ZERO;
      } else if (Value == 1) {
        ImmReg = AMDGPU::ONE_INT;
      } else {
        ImmValue = Value;
      }
    }

    // Check that we aren't already using an immediate.
    // XXX: It's possible for an instruction to have more than one
    // immediate operand, but this is not supported yet.
    if (ImmReg == AMDGPU::ALU_LITERAL_X) {
      if (!Imm.getNode())
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(Imm);
      assert(C);
      if (C->getZExtValue())
        return false;
      Imm = DAG.getTargetConstant(ImmValue, SDLoc(ParentNode), MVT::i32);
    }
    Src = DAG.getRegister(ImmReg, MVT::i32);
    return true;
  }
  default:
    return false;
  }
}


/// \brief Fold the instructions after selecting them
SDNode *R600TargetLowering::PostISelFolding(MachineSDNode *Node,
                                            SelectionDAG &DAG) const {
  const R600InstrInfo *TII =
      static_cast<const R600InstrInfo *>(DAG.getSubtarget().getInstrInfo());
  if (!Node->isMachineOpcode())
    return Node;
  unsigned Opcode = Node->getMachineOpcode();
  SDValue FakeOp;

  std::vector<SDValue> Ops(Node->op_begin(), Node->op_end());

  if (Opcode == AMDGPU::DOT_4) {
    int OperandIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_W),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_W)
        };
    int NegIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_neg_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_neg_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_neg_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_neg_W),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_neg_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_neg_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_neg_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_neg_W)
    };
    int AbsIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_abs_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_abs_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_abs_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_abs_W),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_abs_X),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_abs_Y),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_abs_Z),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_abs_W)
    };
    for (unsigned i = 0; i < 8; i++) {
      if (OperandIdx[i] < 0)
        return Node;
      SDValue &Src = Ops[OperandIdx[i] - 1];
      SDValue &Neg = Ops[NegIdx[i] - 1];
      SDValue &Abs = Ops[AbsIdx[i] - 1];
      bool HasDst = TII->getOperandIdx(Opcode, AMDGPU::OpName::dst) > -1;
      int SelIdx = TII->getSelIdx(Opcode, OperandIdx[i]);
      if (HasDst)
        SelIdx--;
      SDValue &Sel = (SelIdx > -1) ? Ops[SelIdx] : FakeOp;
      if (FoldOperand(Node, i, Src, Neg, Abs, Sel, FakeOp, DAG))
        return DAG.getMachineNode(Opcode, SDLoc(Node), Node->getVTList(), Ops);
    }
  } else if (Opcode == AMDGPU::REG_SEQUENCE) {
    for (unsigned i = 1, e = Node->getNumOperands(); i < e; i += 2) {
      SDValue &Src = Ops[i];
      if (FoldOperand(Node, i, Src, FakeOp, FakeOp, FakeOp, FakeOp, DAG))
        return DAG.getMachineNode(Opcode, SDLoc(Node), Node->getVTList(), Ops);
    }
  } else if (Opcode == AMDGPU::CLAMP_R600) {
    SDValue Src = Node->getOperand(0);
    if (!Src.isMachineOpcode() ||
        !TII->hasInstrModifiers(Src.getMachineOpcode()))
      return Node;
    int ClampIdx = TII->getOperandIdx(Src.getMachineOpcode(),
        AMDGPU::OpName::clamp);
    if (ClampIdx < 0)
      return Node;
    SDLoc DL(Node);
    std::vector<SDValue> Ops(Src->op_begin(), Src->op_end());
    Ops[ClampIdx - 1] = DAG.getTargetConstant(1, DL, MVT::i32);
    return DAG.getMachineNode(Src.getMachineOpcode(), DL,
                              Node->getVTList(), Ops);
  } else {
    if (!TII->hasInstrModifiers(Opcode))
      return Node;
    int OperandIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src2)
    };
    int NegIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_neg),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_neg),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src2_neg)
    };
    int AbsIdx[] = {
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src0_abs),
      TII->getOperandIdx(Opcode, AMDGPU::OpName::src1_abs),
      -1
    };
    for (unsigned i = 0; i < 3; i++) {
      if (OperandIdx[i] < 0)
        return Node;
      SDValue &Src = Ops[OperandIdx[i] - 1];
      SDValue &Neg = Ops[NegIdx[i] - 1];
      SDValue FakeAbs;
      SDValue &Abs = (AbsIdx[i] > -1) ? Ops[AbsIdx[i] - 1] : FakeAbs;
      bool HasDst = TII->getOperandIdx(Opcode, AMDGPU::OpName::dst) > -1;
      int SelIdx = TII->getSelIdx(Opcode, OperandIdx[i]);
      int ImmIdx = TII->getOperandIdx(Opcode, AMDGPU::OpName::literal);
      if (HasDst) {
        SelIdx--;
        ImmIdx--;
      }
      SDValue &Sel = (SelIdx > -1) ? Ops[SelIdx] : FakeOp;
      SDValue &Imm = Ops[ImmIdx];
      if (FoldOperand(Node, i, Src, Neg, Abs, Sel, Imm, DAG))
        return DAG.getMachineNode(Opcode, SDLoc(Node), Node->getVTList(), Ops);
    }
  }

  return Node;
}
