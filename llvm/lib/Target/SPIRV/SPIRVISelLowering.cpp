//===- SPIRVISelLowering.cpp - SPIR-V DAG Lowering Impl ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIRVTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "SPIRVISelLowering.h"
#include "SPIRV.h"
#include "SPIRVInstrInfo.h"
#include "SPIRVRegisterBankInfo.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#define DEBUG_TYPE "spirv-lower"

using namespace llvm;

unsigned SPIRVTargetLowering::getNumRegistersForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return 1 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3 &&
      (VT.getVectorElementType() == MVT::i1 ||
       VT.getVectorElementType() == MVT::i8))
    return 1;
  if (!VT.isVector() && VT.isInteger() && VT.getSizeInBits() <= 64)
    return 1;
  return getNumRegisters(Context, VT);
}

MVT SPIRVTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                       CallingConv::ID CC,
                                                       EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return i32 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3) {
    if (VT.getVectorElementType() == MVT::i1)
      return MVT::v4i1;
    else if (VT.getVectorElementType() == MVT::i8)
      return MVT::v4i8;
  }
  return getRegisterType(Context, VT);
}

bool SPIRVTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  unsigned AlignIdx = 3;
  switch (Intrinsic) {
  case Intrinsic::spv_load:
    AlignIdx = 2;
    [[fallthrough]];
  case Intrinsic::spv_store: {
    if (I.getNumOperands() >= AlignIdx + 1) {
      auto *AlignOp = cast<ConstantInt>(I.getOperand(AlignIdx));
      Info.align = Align(AlignOp->getZExtValue());
    }
    Info.flags = static_cast<MachineMemOperand::Flags>(
        cast<ConstantInt>(I.getOperand(AlignIdx - 1))->getZExtValue());
    Info.memVT = MVT::i64;
    // TODO: take into account opaque pointers (don't use getElementType).
    // MVT::getVT(PtrTy->getElementType());
    return true;
    break;
  }
  default:
    break;
  }
  return false;
}

// Insert a bitcast before the instruction to keep SPIR-V code valid
// when there is a type mismatch between results and operand types.
static void validatePtrTypes(const SPIRVSubtarget &STI,
                             MachineRegisterInfo *MRI, SPIRVGlobalRegistry &GR,
                             MachineInstr &I, SPIRVType *ResType,
                             unsigned OpIdx) {
  Register OpReg = I.getOperand(OpIdx).getReg();
  SPIRVType *TypeInst = MRI->getVRegDef(OpReg);
  SPIRVType *OpType = GR.getSPIRVTypeForVReg(
      TypeInst && TypeInst->getOpcode() == SPIRV::OpFunctionParameter
          ? TypeInst->getOperand(1).getReg()
          : OpReg);
  if (!ResType || !OpType || OpType->getOpcode() != SPIRV::OpTypePointer)
    return;
  SPIRVType *ElemType = GR.getSPIRVTypeForVReg(OpType->getOperand(2).getReg());
  if (!ElemType || ElemType == ResType)
    return;
  // There is a type mismatch between results and operand types
  // and we insert a bitcast before the instruction to keep SPIR-V code valid
  SPIRV::StorageClass::StorageClass SC =
      static_cast<SPIRV::StorageClass::StorageClass>(
          OpType->getOperand(1).getImm());
  MachineIRBuilder MIB(I);
  SPIRVType *NewPtrType = GR.getOrCreateSPIRVPointerType(ResType, MIB, SC);
  if (!GR.isBitcastCompatible(NewPtrType, OpType))
    report_fatal_error(
        "insert validation bitcast: incompatible result and operand types");
  Register NewReg = MRI->createGenericVirtualRegister(LLT::scalar(32));
  bool Res = MIB.buildInstr(SPIRV::OpBitcast)
                 .addDef(NewReg)
                 .addUse(GR.getSPIRVTypeID(NewPtrType))
                 .addUse(OpReg)
                 .constrainAllUses(*STI.getInstrInfo(), *STI.getRegisterInfo(),
                                   *STI.getRegBankInfo());
  if (!Res)
    report_fatal_error("insert validation bitcast: cannot constrain all uses");
  MRI->setRegClass(NewReg, &SPIRV::IDRegClass);
  GR.assignSPIRVTypeToVReg(NewPtrType, NewReg, MIB.getMF());
  I.getOperand(OpIdx).setReg(NewReg);
}

// TODO: the logic of inserting additional bitcast's is to be moved
// to pre-IRTranslation passes eventually
void SPIRVTargetLowering::finalizeLowering(MachineFunction &MF) const {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  SPIRVGlobalRegistry &GR = *STI.getSPIRVGlobalRegistry();
  GR.setCurrentFunc(MF);
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE;) {
      MachineInstr &MI = *MBBI++;
      switch (MI.getOpcode()) {
      case SPIRV::OpLoad:
        // OpLoad <ResType>, ptr %Op implies that %Op is a pointer to <ResType>
        validatePtrTypes(STI, MRI, GR, MI,
                         GR.getSPIRVTypeForVReg(MI.getOperand(0).getReg()), 2);
        break;
      case SPIRV::OpStore:
        // OpStore ptr %Op, <Obj> implies that %Op points to the <Obj>'s type
        validatePtrTypes(STI, MRI, GR, MI,
                         GR.getSPIRVTypeForVReg(MI.getOperand(1).getReg()), 0);
        break;
      // ensure that LLVM IR bitwise instructions result in logical SPIR-V
      // instructions when applied to bool type
      case SPIRV::OpBitwiseOrS:
      case SPIRV::OpBitwiseOrV:
        if (GR.isScalarOrVectorOfType(MI.getOperand(1).getReg(),
                                      SPIRV::OpTypeBool))
          MI.setDesc(STI.getInstrInfo()->get(SPIRV::OpLogicalOr));
        break;
      case SPIRV::OpBitwiseAndS:
      case SPIRV::OpBitwiseAndV:
        if (GR.isScalarOrVectorOfType(MI.getOperand(1).getReg(),
                                      SPIRV::OpTypeBool))
          MI.setDesc(STI.getInstrInfo()->get(SPIRV::OpLogicalAnd));
        break;
      case SPIRV::OpBitwiseXorS:
      case SPIRV::OpBitwiseXorV:
        if (GR.isScalarOrVectorOfType(MI.getOperand(1).getReg(),
                                      SPIRV::OpTypeBool))
          MI.setDesc(STI.getInstrInfo()->get(SPIRV::OpLogicalNotEqual));
        break;
      }
    }
  }
  TargetLowering::finalizeLowering(MF);
}
