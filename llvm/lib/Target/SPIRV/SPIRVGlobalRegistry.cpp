//===-- SPIRVGlobalRegistry.cpp - SPIR-V Global Registry --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the SPIRVGlobalRegistry class,
// which is used to maintain rich type information required for SPIR-V even
// after lowering from LLVM IR to GMIR. It can convert an llvm::Type into
// an OpTypeXXX instruction, and map it to a virtual register. Also it builds
// and supports consistency of constants and global variables.
//
//===----------------------------------------------------------------------===//

#include "SPIRVGlobalRegistry.h"
#include "SPIRV.h"
#include "SPIRVBuiltins.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <functional>

using namespace llvm;

static bool allowEmitFakeUse(const Value *Arg) {
  if (isSpvIntrinsic(Arg))
    return false;
  if (isa<AtomicCmpXchgInst, InsertValueInst, UndefValue>(Arg))
    return false;
  if (const auto *LI = dyn_cast<LoadInst>(Arg))
    if (LI->getType()->isAggregateType())
      return false;
  return true;
}

static unsigned typeToAddressSpace(const Type *Ty) {
  if (auto PType = dyn_cast<TypedPointerType>(Ty))
    return PType->getAddressSpace();
  if (auto PType = dyn_cast<PointerType>(Ty))
    return PType->getAddressSpace();
  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty);
      ExtTy && isTypedPointerWrapper(ExtTy))
    return ExtTy->getIntParameter(0);
  reportFatalInternalError("Unable to convert LLVM type to SPIRVType");
}

static bool
storageClassRequiresExplictLayout(SPIRV::StorageClass::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::Uniform:
  case SPIRV::StorageClass::PushConstant:
  case SPIRV::StorageClass::StorageBuffer:
  case SPIRV::StorageClass::PhysicalStorageBufferEXT:
    return true;
  case SPIRV::StorageClass::UniformConstant:
  case SPIRV::StorageClass::Input:
  case SPIRV::StorageClass::Output:
  case SPIRV::StorageClass::Workgroup:
  case SPIRV::StorageClass::CrossWorkgroup:
  case SPIRV::StorageClass::Private:
  case SPIRV::StorageClass::Function:
  case SPIRV::StorageClass::Generic:
  case SPIRV::StorageClass::AtomicCounter:
  case SPIRV::StorageClass::Image:
  case SPIRV::StorageClass::CallableDataNV:
  case SPIRV::StorageClass::IncomingCallableDataNV:
  case SPIRV::StorageClass::RayPayloadNV:
  case SPIRV::StorageClass::HitAttributeNV:
  case SPIRV::StorageClass::IncomingRayPayloadNV:
  case SPIRV::StorageClass::ShaderRecordBufferNV:
  case SPIRV::StorageClass::CodeSectionINTEL:
  case SPIRV::StorageClass::DeviceOnlyINTEL:
  case SPIRV::StorageClass::HostOnlyINTEL:
    return false;
  }
  llvm_unreachable("Unknown SPIRV::StorageClass enum");
}

SPIRVGlobalRegistry::SPIRVGlobalRegistry(unsigned PointerSize)
    : PointerSize(PointerSize), Bound(0) {}

SPIRVType *SPIRVGlobalRegistry::assignIntTypeToVReg(unsigned BitWidth,
                                                    Register VReg,
                                                    MachineInstr &I,
                                                    const SPIRVInstrInfo &TII) {
  SPIRVType *SpirvType = getOrCreateSPIRVIntegerType(BitWidth, I, TII);
  assignSPIRVTypeToVReg(SpirvType, VReg, *CurMF);
  return SpirvType;
}

SPIRVType *
SPIRVGlobalRegistry::assignFloatTypeToVReg(unsigned BitWidth, Register VReg,
                                           MachineInstr &I,
                                           const SPIRVInstrInfo &TII) {
  SPIRVType *SpirvType = getOrCreateSPIRVFloatType(BitWidth, I, TII);
  assignSPIRVTypeToVReg(SpirvType, VReg, *CurMF);
  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::assignVectTypeToVReg(
    SPIRVType *BaseType, unsigned NumElements, Register VReg, MachineInstr &I,
    const SPIRVInstrInfo &TII) {
  SPIRVType *SpirvType =
      getOrCreateSPIRVVectorType(BaseType, NumElements, I, TII);
  assignSPIRVTypeToVReg(SpirvType, VReg, *CurMF);
  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::assignTypeToVReg(
    const Type *Type, Register VReg, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual, bool EmitIR) {
  SPIRVType *SpirvType =
      getOrCreateSPIRVType(Type, MIRBuilder, AccessQual, EmitIR);
  assignSPIRVTypeToVReg(SpirvType, VReg, MIRBuilder.getMF());
  return SpirvType;
}

void SPIRVGlobalRegistry::assignSPIRVTypeToVReg(SPIRVType *SpirvType,
                                                Register VReg,
                                                const MachineFunction &MF) {
  VRegToTypeMap[&MF][VReg] = SpirvType;
}

static Register createTypeVReg(MachineRegisterInfo &MRI) {
  auto Res = MRI.createGenericVirtualRegister(LLT::scalar(64));
  MRI.setRegClass(Res, &SPIRV::TYPERegClass);
  return Res;
}

inline Register createTypeVReg(MachineIRBuilder &MIRBuilder) {
  return createTypeVReg(MIRBuilder.getMF().getRegInfo());
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeBool(MachineIRBuilder &MIRBuilder) {
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypeBool)
        .addDef(createTypeVReg(MIRBuilder));
  });
}

unsigned SPIRVGlobalRegistry::adjustOpTypeIntWidth(unsigned Width) const {
  if (Width > 64)
    report_fatal_error("Unsupported integer width!");
  const SPIRVSubtarget &ST = cast<SPIRVSubtarget>(CurMF->getSubtarget());
  if (ST.canUseExtension(
          SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers) ||
      ST.canUseExtension(SPIRV::Extension::SPV_INTEL_int4))
    return Width;
  if (Width <= 8)
    Width = 8;
  else if (Width <= 16)
    Width = 16;
  else if (Width <= 32)
    Width = 32;
  else
    Width = 64;
  return Width;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeInt(unsigned Width,
                                             MachineIRBuilder &MIRBuilder,
                                             bool IsSigned) {
  Width = adjustOpTypeIntWidth(Width);
  const SPIRVSubtarget &ST =
      cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    if (Width == 4 && ST.canUseExtension(SPIRV::Extension::SPV_INTEL_int4)) {
      MIRBuilder.buildInstr(SPIRV::OpExtension)
          .addImm(SPIRV::Extension::SPV_INTEL_int4);
      MIRBuilder.buildInstr(SPIRV::OpCapability)
          .addImm(SPIRV::Capability::Int4TypeINTEL);
    } else if ((!isPowerOf2_32(Width) || Width < 8) &&
               ST.canUseExtension(
                   SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers)) {
      MIRBuilder.buildInstr(SPIRV::OpExtension)
          .addImm(SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers);
      MIRBuilder.buildInstr(SPIRV::OpCapability)
          .addImm(SPIRV::Capability::ArbitraryPrecisionIntegersINTEL);
    }
    return MIRBuilder.buildInstr(SPIRV::OpTypeInt)
        .addDef(createTypeVReg(MIRBuilder))
        .addImm(Width)
        .addImm(IsSigned ? 1 : 0);
  });
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeFloat(uint32_t Width,
                                               MachineIRBuilder &MIRBuilder) {
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypeFloat)
        .addDef(createTypeVReg(MIRBuilder))
        .addImm(Width);
  });
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeVoid(MachineIRBuilder &MIRBuilder) {
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypeVoid)
        .addDef(createTypeVReg(MIRBuilder));
  });
}

void SPIRVGlobalRegistry::invalidateMachineInstr(MachineInstr *MI) {
  // TODO:
  // - review other data structure wrt. possible issues related to removal
  //   of a machine instruction during instruction selection.
  const MachineFunction *MF = MI->getMF();
  auto It = LastInsertedTypeMap.find(MF);
  if (It == LastInsertedTypeMap.end())
    return;
  if (It->second == MI)
    LastInsertedTypeMap.erase(MF);
  // remove from the duplicate tracker to avoid incorrect reuse
  erase(MI);
}

SPIRVType *SPIRVGlobalRegistry::createOpType(
    MachineIRBuilder &MIRBuilder,
    std::function<MachineInstr *(MachineIRBuilder &)> Op) {
  auto oldInsertPoint = MIRBuilder.getInsertPt();
  MachineBasicBlock *OldMBB = &MIRBuilder.getMBB();
  MachineBasicBlock *NewMBB = &*MIRBuilder.getMF().begin();

  auto LastInsertedType = LastInsertedTypeMap.find(CurMF);
  if (LastInsertedType != LastInsertedTypeMap.end()) {
    auto It = LastInsertedType->second->getIterator();
    // It might happen that this instruction was removed from the first MBB,
    // hence the Parent's check.
    MachineBasicBlock::iterator InsertAt;
    if (It->getParent() != NewMBB)
      InsertAt = oldInsertPoint->getParent() == NewMBB
                     ? oldInsertPoint
                     : getInsertPtValidEnd(NewMBB);
    else if (It->getNextNode())
      InsertAt = It->getNextNode()->getIterator();
    else
      InsertAt = getInsertPtValidEnd(NewMBB);
    MIRBuilder.setInsertPt(*NewMBB, InsertAt);
  } else {
    MIRBuilder.setInsertPt(*NewMBB, NewMBB->begin());
    auto Result = LastInsertedTypeMap.try_emplace(CurMF, nullptr);
    assert(Result.second);
    LastInsertedType = Result.first;
  }

  MachineInstr *Type = Op(MIRBuilder);
  // We expect all users of this function to insert definitions at the insertion
  // point set above that is always the first MBB.
  assert(Type->getParent() == NewMBB);
  LastInsertedType->second = Type;

  MIRBuilder.setInsertPt(*OldMBB, oldInsertPoint);
  return Type;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeVector(uint32_t NumElems,
                                                SPIRVType *ElemType,
                                                MachineIRBuilder &MIRBuilder) {
  auto EleOpc = ElemType->getOpcode();
  (void)EleOpc;
  assert((EleOpc == SPIRV::OpTypeInt || EleOpc == SPIRV::OpTypeFloat ||
          EleOpc == SPIRV::OpTypeBool) &&
         "Invalid vector element type");

  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypeVector)
        .addDef(createTypeVReg(MIRBuilder))
        .addUse(getSPIRVTypeID(ElemType))
        .addImm(NumElems);
  });
}

Register SPIRVGlobalRegistry::getOrCreateConstFP(APFloat Val, MachineInstr &I,
                                                 SPIRVType *SpvType,
                                                 const SPIRVInstrInfo &TII,
                                                 bool ZeroAsNull) {
  LLVMContext &Ctx = CurMF->getFunction().getContext();
  auto *const CF = ConstantFP::get(Ctx, Val);
  const MachineInstr *MI = findMI(CF, CurMF);
  if (MI && (MI->getOpcode() == SPIRV::OpConstantNull ||
             MI->getOpcode() == SPIRV::OpConstantF))
    return MI->getOperand(0).getReg();
  return createConstFP(CF, I, SpvType, TII, ZeroAsNull);
}

Register SPIRVGlobalRegistry::createConstFP(const ConstantFP *CF,
                                            MachineInstr &I, SPIRVType *SpvType,
                                            const SPIRVInstrInfo &TII,
                                            bool ZeroAsNull) {
  unsigned BitWidth = getScalarOrVectorBitWidth(SpvType);
  LLT LLTy = LLT::scalar(BitWidth);
  Register Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::fIDRegClass);
  assignFloatTypeToVReg(BitWidth, Res, I, TII);

  MachineInstr *DepMI = const_cast<MachineInstr *>(SpvType);
  MachineIRBuilder MIRBuilder(*DepMI->getParent(), DepMI->getIterator());
  SPIRVType *NewType =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        MachineInstrBuilder MIB;
        // In OpenCL OpConstantNull - Scalar floating point: +0.0 (all bits 0)
        if (CF->getValue().isPosZero() && ZeroAsNull) {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantNull)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
        } else {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantF)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
          addNumImm(APInt(BitWidth,
                          CF->getValueAPF().bitcastToAPInt().getZExtValue()),
                    MIB);
        }
        const auto &ST = CurMF->getSubtarget();
        constrainSelectedInstRegOperands(*MIB, *ST.getInstrInfo(),
                                         *ST.getRegisterInfo(),
                                         *ST.getRegBankInfo());
        return MIB;
      });
  add(CF, NewType);
  return Res;
}

Register SPIRVGlobalRegistry::getOrCreateConstInt(uint64_t Val, MachineInstr &I,
                                                  SPIRVType *SpvType,
                                                  const SPIRVInstrInfo &TII,
                                                  bool ZeroAsNull) {
  const IntegerType *Ty = cast<IntegerType>(getTypeForSPIRVType(SpvType));
  auto *const CI = ConstantInt::get(const_cast<IntegerType *>(Ty), Val);
  const MachineInstr *MI = findMI(CI, CurMF);
  if (MI && (MI->getOpcode() == SPIRV::OpConstantNull ||
             MI->getOpcode() == SPIRV::OpConstantI))
    return MI->getOperand(0).getReg();
  return createConstInt(CI, I, SpvType, TII, ZeroAsNull);
}

Register SPIRVGlobalRegistry::createConstInt(const ConstantInt *CI,
                                             MachineInstr &I,
                                             SPIRVType *SpvType,
                                             const SPIRVInstrInfo &TII,
                                             bool ZeroAsNull) {
  unsigned BitWidth = getScalarOrVectorBitWidth(SpvType);
  LLT LLTy = LLT::scalar(BitWidth);
  Register Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::iIDRegClass);
  assignIntTypeToVReg(BitWidth, Res, I, TII);

  MachineInstr *DepMI = const_cast<MachineInstr *>(SpvType);
  MachineIRBuilder MIRBuilder(*DepMI->getParent(), DepMI->getIterator());
  SPIRVType *NewType =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        MachineInstrBuilder MIB;
        if (BitWidth == 1) {
          MIB = MIRBuilder
                    .buildInstr(CI->isZero() ? SPIRV::OpConstantFalse
                                             : SPIRV::OpConstantTrue)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
        } else if (!CI->isZero() || !ZeroAsNull) {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantI)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
          addNumImm(APInt(BitWidth, CI->getZExtValue()), MIB);
        } else {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantNull)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
        }
        const auto &ST = CurMF->getSubtarget();
        constrainSelectedInstRegOperands(*MIB, *ST.getInstrInfo(),
                                         *ST.getRegisterInfo(),
                                         *ST.getRegBankInfo());
        return MIB;
      });
  add(CI, NewType);
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantInt(uint64_t Val,
                                               MachineIRBuilder &MIRBuilder,
                                               SPIRVType *SpvType, bool EmitIR,
                                               bool ZeroAsNull) {
  assert(SpvType);
  auto &MF = MIRBuilder.getMF();
  const IntegerType *Ty = cast<IntegerType>(getTypeForSPIRVType(SpvType));
  auto *const CI = ConstantInt::get(const_cast<IntegerType *>(Ty), Val);
  Register Res = find(CI, &MF);
  if (Res.isValid())
    return Res;

  unsigned BitWidth = getScalarOrVectorBitWidth(SpvType);
  LLT LLTy = LLT::scalar(BitWidth);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Res = MRI.createGenericVirtualRegister(LLTy);
  MRI.setRegClass(Res, &SPIRV::iIDRegClass);
  assignTypeToVReg(Ty, Res, MIRBuilder, SPIRV::AccessQualifier::ReadWrite,
                   EmitIR);

  SPIRVType *NewType =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        if (EmitIR)
          return MIRBuilder.buildConstant(Res, *CI);
        Register SpvTypeReg = getSPIRVTypeID(SpvType);
        MachineInstrBuilder MIB;
        if (Val || !ZeroAsNull) {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantI)
                    .addDef(Res)
                    .addUse(SpvTypeReg);
          addNumImm(APInt(BitWidth, Val), MIB);
        } else {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantNull)
                    .addDef(Res)
                    .addUse(SpvTypeReg);
        }
        const auto &Subtarget = CurMF->getSubtarget();
        constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                         *Subtarget.getRegisterInfo(),
                                         *Subtarget.getRegBankInfo());
        return MIB;
      });
  add(CI, NewType);
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantFP(APFloat Val,
                                              MachineIRBuilder &MIRBuilder,
                                              SPIRVType *SpvType) {
  auto &MF = MIRBuilder.getMF();
  LLVMContext &Ctx = MF.getFunction().getContext();
  if (!SpvType)
    SpvType = getOrCreateSPIRVType(Type::getFloatTy(Ctx), MIRBuilder,
                                   SPIRV::AccessQualifier::ReadWrite, true);
  auto *const CF = ConstantFP::get(Ctx, Val);
  Register Res = find(CF, &MF);
  if (Res.isValid())
    return Res;

  LLT LLTy = LLT::scalar(getScalarOrVectorBitWidth(SpvType));
  Res = MF.getRegInfo().createGenericVirtualRegister(LLTy);
  MF.getRegInfo().setRegClass(Res, &SPIRV::fIDRegClass);
  assignSPIRVTypeToVReg(SpvType, Res, MF);

  SPIRVType *NewType =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        MachineInstrBuilder MIB;
        MIB = MIRBuilder.buildInstr(SPIRV::OpConstantF)
                  .addDef(Res)
                  .addUse(getSPIRVTypeID(SpvType));
        addNumImm(CF->getValueAPF().bitcastToAPInt(), MIB);
        return MIB;
      });
  add(CF, NewType);
  return Res;
}

Register SPIRVGlobalRegistry::getOrCreateBaseRegister(
    Constant *Val, MachineInstr &I, SPIRVType *SpvType,
    const SPIRVInstrInfo &TII, unsigned BitWidth, bool ZeroAsNull) {
  SPIRVType *Type = SpvType;
  if (SpvType->getOpcode() == SPIRV::OpTypeVector ||
      SpvType->getOpcode() == SPIRV::OpTypeArray) {
    auto EleTypeReg = SpvType->getOperand(1).getReg();
    Type = getSPIRVTypeForVReg(EleTypeReg);
  }
  if (Type->getOpcode() == SPIRV::OpTypeFloat) {
    SPIRVType *SpvBaseType = getOrCreateSPIRVFloatType(BitWidth, I, TII);
    return getOrCreateConstFP(dyn_cast<ConstantFP>(Val)->getValue(), I,
                              SpvBaseType, TII, ZeroAsNull);
  }
  assert(Type->getOpcode() == SPIRV::OpTypeInt);
  SPIRVType *SpvBaseType = getOrCreateSPIRVIntegerType(BitWidth, I, TII);
  return getOrCreateConstInt(Val->getUniqueInteger().getZExtValue(), I,
                             SpvBaseType, TII, ZeroAsNull);
}

Register SPIRVGlobalRegistry::getOrCreateCompositeOrNull(
    Constant *Val, MachineInstr &I, SPIRVType *SpvType,
    const SPIRVInstrInfo &TII, Constant *CA, unsigned BitWidth,
    unsigned ElemCnt, bool ZeroAsNull) {
  if (Register R = find(CA, CurMF); R.isValid())
    return R;

  bool IsNull = Val->isNullValue() && ZeroAsNull;
  Register ElemReg;
  if (!IsNull)
    ElemReg =
        getOrCreateBaseRegister(Val, I, SpvType, TII, BitWidth, ZeroAsNull);

  LLT LLTy = LLT::scalar(64);
  Register Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, getRegClass(SpvType));
  assignSPIRVTypeToVReg(SpvType, Res, *CurMF);

  MachineInstr *DepMI = const_cast<MachineInstr *>(SpvType);
  MachineIRBuilder MIRBuilder(*DepMI->getParent(), DepMI->getIterator());
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        MachineInstrBuilder MIB;
        if (!IsNull) {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantComposite)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
          for (unsigned i = 0; i < ElemCnt; ++i)
            MIB.addUse(ElemReg);
        } else {
          MIB = MIRBuilder.buildInstr(SPIRV::OpConstantNull)
                    .addDef(Res)
                    .addUse(getSPIRVTypeID(SpvType));
        }
        const auto &Subtarget = CurMF->getSubtarget();
        constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                         *Subtarget.getRegisterInfo(),
                                         *Subtarget.getRegBankInfo());
        return MIB;
      });
  add(CA, NewMI);
  return Res;
}

Register SPIRVGlobalRegistry::getOrCreateConstVector(uint64_t Val,
                                                     MachineInstr &I,
                                                     SPIRVType *SpvType,
                                                     const SPIRVInstrInfo &TII,
                                                     bool ZeroAsNull) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isVectorTy());
  const FixedVectorType *LLVMVecTy = cast<FixedVectorType>(LLVMTy);
  Type *LLVMBaseTy = LLVMVecTy->getElementType();
  assert(LLVMBaseTy->isIntegerTy());
  auto *ConstVal = ConstantInt::get(LLVMBaseTy, Val);
  auto *ConstVec =
      ConstantVector::getSplat(LLVMVecTy->getElementCount(), ConstVal);
  unsigned BW = getScalarOrVectorBitWidth(SpvType);
  return getOrCreateCompositeOrNull(ConstVal, I, SpvType, TII, ConstVec, BW,
                                    SpvType->getOperand(2).getImm(),
                                    ZeroAsNull);
}

Register SPIRVGlobalRegistry::getOrCreateConstVector(APFloat Val,
                                                     MachineInstr &I,
                                                     SPIRVType *SpvType,
                                                     const SPIRVInstrInfo &TII,
                                                     bool ZeroAsNull) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isVectorTy());
  const FixedVectorType *LLVMVecTy = cast<FixedVectorType>(LLVMTy);
  Type *LLVMBaseTy = LLVMVecTy->getElementType();
  assert(LLVMBaseTy->isFloatingPointTy());
  auto *ConstVal = ConstantFP::get(LLVMBaseTy, Val);
  auto *ConstVec =
      ConstantVector::getSplat(LLVMVecTy->getElementCount(), ConstVal);
  unsigned BW = getScalarOrVectorBitWidth(SpvType);
  return getOrCreateCompositeOrNull(ConstVal, I, SpvType, TII, ConstVec, BW,
                                    SpvType->getOperand(2).getImm(),
                                    ZeroAsNull);
}

Register SPIRVGlobalRegistry::getOrCreateConstIntArray(
    uint64_t Val, size_t Num, MachineInstr &I, SPIRVType *SpvType,
    const SPIRVInstrInfo &TII) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isArrayTy());
  const ArrayType *LLVMArrTy = cast<ArrayType>(LLVMTy);
  Type *LLVMBaseTy = LLVMArrTy->getElementType();
  Constant *CI = ConstantInt::get(LLVMBaseTy, Val);
  SPIRVType *SpvBaseTy = getSPIRVTypeForVReg(SpvType->getOperand(1).getReg());
  unsigned BW = getScalarOrVectorBitWidth(SpvBaseTy);
  // The following is reasonably unique key that is better that [Val]. The naive
  // alternative would be something along the lines of:
  //   SmallVector<Constant *> NumCI(Num, CI);
  //   Constant *UniqueKey =
  //     ConstantArray::get(const_cast<ArrayType*>(LLVMArrTy), NumCI);
  // that would be a truly unique but dangerous key, because it could lead to
  // the creation of constants of arbitrary length (that is, the parameter of
  // memset) which were missing in the original module.
  Constant *UniqueKey = ConstantStruct::getAnon(
      {PoisonValue::get(const_cast<ArrayType *>(LLVMArrTy)),
       ConstantInt::get(LLVMBaseTy, Val), ConstantInt::get(LLVMBaseTy, Num)});
  return getOrCreateCompositeOrNull(CI, I, SpvType, TII, UniqueKey, BW,
                                    LLVMArrTy->getNumElements());
}

Register SPIRVGlobalRegistry::getOrCreateIntCompositeOrNull(
    uint64_t Val, MachineIRBuilder &MIRBuilder, SPIRVType *SpvType, bool EmitIR,
    Constant *CA, unsigned BitWidth, unsigned ElemCnt) {
  if (Register R = find(CA, CurMF); R.isValid())
    return R;

  Register ElemReg;
  if (Val || EmitIR) {
    SPIRVType *SpvBaseType = getOrCreateSPIRVIntegerType(BitWidth, MIRBuilder);
    ElemReg = buildConstantInt(Val, MIRBuilder, SpvBaseType, EmitIR);
  }
  LLT LLTy = EmitIR ? LLT::fixed_vector(ElemCnt, BitWidth) : LLT::scalar(64);
  Register Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::iIDRegClass);
  assignSPIRVTypeToVReg(SpvType, Res, *CurMF);

  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        if (EmitIR)
          return MIRBuilder.buildSplatBuildVector(Res, ElemReg);

        if (Val) {
          auto MIB = MIRBuilder.buildInstr(SPIRV::OpConstantComposite)
                         .addDef(Res)
                         .addUse(getSPIRVTypeID(SpvType));
          for (unsigned i = 0; i < ElemCnt; ++i)
            MIB.addUse(ElemReg);
          return MIB;
        }

        return MIRBuilder.buildInstr(SPIRV::OpConstantNull)
            .addDef(Res)
            .addUse(getSPIRVTypeID(SpvType));
      });
  add(CA, NewMI);
  return Res;
}

Register
SPIRVGlobalRegistry::getOrCreateConsIntVector(uint64_t Val,
                                              MachineIRBuilder &MIRBuilder,
                                              SPIRVType *SpvType, bool EmitIR) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isVectorTy());
  const FixedVectorType *LLVMVecTy = cast<FixedVectorType>(LLVMTy);
  Type *LLVMBaseTy = LLVMVecTy->getElementType();
  const auto ConstInt = ConstantInt::get(LLVMBaseTy, Val);
  auto ConstVec =
      ConstantVector::getSplat(LLVMVecTy->getElementCount(), ConstInt);
  unsigned BW = getScalarOrVectorBitWidth(SpvType);
  return getOrCreateIntCompositeOrNull(Val, MIRBuilder, SpvType, EmitIR,
                                       ConstVec, BW,
                                       SpvType->getOperand(2).getImm());
}

Register
SPIRVGlobalRegistry::getOrCreateConstNullPtr(MachineIRBuilder &MIRBuilder,
                                             SPIRVType *SpvType) {
  const Type *Ty = getTypeForSPIRVType(SpvType);
  unsigned AddressSpace = typeToAddressSpace(Ty);
  Type *ElemTy = ::getPointeeType(Ty);
  assert(ElemTy);
  const Constant *CP = ConstantTargetNone::get(
      dyn_cast<TargetExtType>(getTypedPointerWrapper(ElemTy, AddressSpace)));
  Register Res = find(CP, CurMF);
  if (Res.isValid())
    return Res;

  LLT LLTy = LLT::pointer(AddressSpace, PointerSize);
  Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::pIDRegClass);
  assignSPIRVTypeToVReg(SpvType, Res, *CurMF);

  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(SPIRV::OpConstantNull)
            .addDef(Res)
            .addUse(getSPIRVTypeID(SpvType));
      });
  add(CP, NewMI);
  return Res;
}

Register
SPIRVGlobalRegistry::buildConstantSampler(Register ResReg, unsigned AddrMode,
                                          unsigned Param, unsigned FilerMode,
                                          MachineIRBuilder &MIRBuilder) {
  auto Sampler =
      ResReg.isValid()
          ? ResReg
          : MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::iIDRegClass);
  SPIRVType *TypeSampler = getOrCreateOpTypeSampler(MIRBuilder);
  Register TypeSamplerReg = getSPIRVTypeID(TypeSampler);
  // We cannot use createOpType() logic here, because of the
  // GlobalISel/IRTranslator.cpp check for a tail call that expects that
  // MIRBuilder.getInsertPt() has a previous instruction. If this constant is
  // inserted as a result of "__translate_sampler_initializer()" this would
  // break this IRTranslator assumption.
  MIRBuilder.buildInstr(SPIRV::OpConstantSampler)
      .addDef(Sampler)
      .addUse(TypeSamplerReg)
      .addImm(AddrMode)
      .addImm(Param)
      .addImm(FilerMode);
  return Sampler;
}

Register SPIRVGlobalRegistry::buildGlobalVariable(
    Register ResVReg, SPIRVType *BaseType, StringRef Name,
    const GlobalValue *GV, SPIRV::StorageClass::StorageClass Storage,
    const MachineInstr *Init, bool IsConst, bool HasLinkageTy,
    SPIRV::LinkageType::LinkageType LinkageType, MachineIRBuilder &MIRBuilder,
    bool IsInstSelector) {
  const GlobalVariable *GVar = nullptr;
  if (GV) {
    GVar = cast<const GlobalVariable>(GV);
  } else {
    // If GV is not passed explicitly, use the name to find or construct
    // the global variable.
    Module *M = MIRBuilder.getMF().getFunction().getParent();
    GVar = M->getGlobalVariable(Name);
    if (GVar == nullptr) {
      const Type *Ty = getTypeForSPIRVType(BaseType); // TODO: check type.
      // Module takes ownership of the global var.
      GVar = new GlobalVariable(*M, const_cast<Type *>(Ty), false,
                                GlobalValue::ExternalLinkage, nullptr,
                                Twine(Name));
    }
    GV = GVar;
  }

  const MachineFunction *MF = &MIRBuilder.getMF();
  Register Reg = find(GVar, MF);
  if (Reg.isValid()) {
    if (Reg != ResVReg)
      MIRBuilder.buildCopy(ResVReg, Reg);
    return ResVReg;
  }

  auto MIB = MIRBuilder.buildInstr(SPIRV::OpVariable)
                 .addDef(ResVReg)
                 .addUse(getSPIRVTypeID(BaseType))
                 .addImm(static_cast<uint32_t>(Storage));
  if (Init != 0)
    MIB.addUse(Init->getOperand(0).getReg());
  // ISel may introduce a new register on this step, so we need to add it to
  // DT and correct its type avoiding fails on the next stage.
  if (IsInstSelector) {
    const auto &Subtarget = CurMF->getSubtarget();
    constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                     *Subtarget.getRegisterInfo(),
                                     *Subtarget.getRegBankInfo());
  }
  add(GVar, MIB);

  Reg = MIB->getOperand(0).getReg();
  addGlobalObject(GVar, MF, Reg);

  // Set to Reg the same type as ResVReg has.
  auto MRI = MIRBuilder.getMRI();
  if (Reg != ResVReg) {
    LLT RegLLTy =
        LLT::pointer(MRI->getType(ResVReg).getAddressSpace(), getPointerSize());
    MRI->setType(Reg, RegLLTy);
    assignSPIRVTypeToVReg(BaseType, Reg, MIRBuilder.getMF());
  } else {
    // Our knowledge about the type may be updated.
    // If that's the case, we need to update a type
    // associated with the register.
    SPIRVType *DefType = getSPIRVTypeForVReg(ResVReg);
    if (!DefType || DefType != BaseType)
      assignSPIRVTypeToVReg(BaseType, Reg, MIRBuilder.getMF());
  }

  // If it's a global variable with name, output OpName for it.
  if (GVar && GVar->hasName())
    buildOpName(Reg, GVar->getName(), MIRBuilder);

  // Output decorations for the GV.
  // TODO: maybe move to GenerateDecorations pass.
  const SPIRVSubtarget &ST =
      cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
  if (IsConst && ST.isOpenCLEnv())
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::Constant, {});

  if (GVar && GVar->getAlign().valueOrOne().value() != 1 && !ST.isVulkanEnv()) {
    unsigned Alignment = (unsigned)GVar->getAlign().valueOrOne().value();
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::Alignment, {Alignment});
  }

  if (HasLinkageTy)
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::LinkageAttributes,
                    {static_cast<uint32_t>(LinkageType)}, Name);

  SPIRV::BuiltIn::BuiltIn BuiltInId;
  if (getSpirvBuiltInIdByName(Name, BuiltInId))
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::BuiltIn,
                    {static_cast<uint32_t>(BuiltInId)});

  // If it's a global variable with "spirv.Decorations" metadata node
  // recognize it as a SPIR-V friendly LLVM IR and parse "spirv.Decorations"
  // arguments.
  MDNode *GVarMD = nullptr;
  if (GVar && (GVarMD = GVar->getMetadata("spirv.Decorations")) != nullptr)
    buildOpSpirvDecorations(Reg, MIRBuilder, GVarMD);

  return Reg;
}

static std::string GetSpirvImageTypeName(const SPIRVType *Type,
                                         MachineIRBuilder &MIRBuilder,
                                         const std::string &Prefix,
                                         SPIRVGlobalRegistry &GR);

// Returns a name based on the Type. Notes that this does not look at
// decorations, and will return the same string for two types that are the same
// except for decorations.
static std::string buildSpirvTypeName(const SPIRVType *Type,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRVGlobalRegistry &GR) {
  switch (Type->getOpcode()) {
  case SPIRV::OpTypeSampledImage: {
    return GetSpirvImageTypeName(Type, MIRBuilder, "sampled_image_", GR);
  }
  case SPIRV::OpTypeImage: {
    return GetSpirvImageTypeName(Type, MIRBuilder, "image_", GR);
  }
  case SPIRV::OpTypeArray: {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    Register ElementTypeReg = Type->getOperand(1).getReg();
    auto *ElementType = MRI->getUniqueVRegDef(ElementTypeReg);
    uint32_t ArraySize = getArrayComponentCount(MRI, Type);
    return (buildSpirvTypeName(ElementType, MIRBuilder, GR) + Twine("[") +
            Twine(ArraySize) + Twine("]"))
        .str();
  }
  case SPIRV::OpTypeFloat:
    return ("f" + Twine(Type->getOperand(1).getImm())).str();
  case SPIRV::OpTypeSampler:
    return ("sampler");
  case SPIRV::OpTypeInt:
    if (Type->getOperand(2).getImm())
      return ("i" + Twine(Type->getOperand(1).getImm())).str();
    return ("u" + Twine(Type->getOperand(1).getImm())).str();
  case SPIRV::OpTypePointer: {
    uint32_t StorageClass = GR.getPointerStorageClass(Type);
    SPIRVType *PointeeType = GR.getPointeeType(Type);
    return ("p_" + Twine(StorageClass) + Twine("_") +
            buildSpirvTypeName(PointeeType, MIRBuilder, GR))
        .str();
  }
  case SPIRV::OpTypeStruct: {
    std::string TypeName = "{";
    for (uint32_t I = 1; I < Type->getNumOperands(); ++I) {
      SPIRVType *MemberType =
          GR.getSPIRVTypeForVReg(Type->getOperand(I).getReg());
      TypeName += '_' + buildSpirvTypeName(MemberType, MIRBuilder, GR);
    }
    return TypeName + "}";
  }
  case SPIRV::OpTypeVector: {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    Register ElementTypeReg = Type->getOperand(1).getReg();
    auto *ElementType = MRI->getUniqueVRegDef(ElementTypeReg);
    uint32_t VectorSize = GR.getScalarOrVectorComponentCount(Type);
    return (buildSpirvTypeName(ElementType, MIRBuilder, GR) + Twine("[") +
            Twine(VectorSize) + Twine("]"))
        .str();
  }
  case SPIRV::OpTypeRuntimeArray: {
    MachineRegisterInfo *MRI = MIRBuilder.getMRI();
    Register ElementTypeReg = Type->getOperand(1).getReg();
    auto *ElementType = MRI->getUniqueVRegDef(ElementTypeReg);
    uint32_t ArraySize = 0;
    return (buildSpirvTypeName(ElementType, MIRBuilder, GR) + Twine("[") +
            Twine(ArraySize) + Twine("]"))
        .str();
  }
  default:
    llvm_unreachable("Trying to the the name of an unknown type.");
  }
}

static std::string GetSpirvImageTypeName(const SPIRVType *Type,
                                         MachineIRBuilder &MIRBuilder,
                                         const std::string &Prefix,
                                         SPIRVGlobalRegistry &GR) {
  Register SampledTypeReg = Type->getOperand(1).getReg();
  auto *SampledType = MIRBuilder.getMRI()->getUniqueVRegDef(SampledTypeReg);
  std::string TypeName =
      Prefix + buildSpirvTypeName(SampledType, MIRBuilder, GR);
  for (uint32_t I = 2; I < Type->getNumOperands(); ++I) {
    TypeName = (TypeName + '_' + Twine(Type->getOperand(I).getImm())).str();
  }
  return TypeName;
}

Register SPIRVGlobalRegistry::getOrCreateGlobalVariableWithBinding(
    const SPIRVType *VarType, uint32_t Set, uint32_t Binding,
    MachineIRBuilder &MIRBuilder) {
  Register VarReg =
      MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::iIDRegClass);

  // TODO(138533): The name should come from the llvm-ir, but how that name will
  // be passed from the HLSL to the backend has not been decided. Using this
  // place holder for now.
  std::string Name =
      ("__resource_" + buildSpirvTypeName(VarType, MIRBuilder, *this) + "_" +
       Twine(Set) + "_" + Twine(Binding))
          .str();
  buildGlobalVariable(VarReg, VarType, Name, nullptr,
                      getPointerStorageClass(VarType), nullptr, false, false,
                      SPIRV::LinkageType::Import, MIRBuilder, false);

  buildOpDecorate(VarReg, MIRBuilder, SPIRV::Decoration::DescriptorSet, {Set});
  buildOpDecorate(VarReg, MIRBuilder, SPIRV::Decoration::Binding, {Binding});
  return VarReg;
}

// TODO: Double check the calls to getOpTypeArray to make sure that `ElemType`
// is explicitly laid out when required.
SPIRVType *SPIRVGlobalRegistry::getOpTypeArray(uint32_t NumElems,
                                               SPIRVType *ElemType,
                                               MachineIRBuilder &MIRBuilder,
                                               bool ExplicitLayoutRequired,
                                               bool EmitIR) {
  assert((ElemType->getOpcode() != SPIRV::OpTypeVoid) &&
         "Invalid array element type");
  SPIRVType *SpvTypeInt32 = getOrCreateSPIRVIntegerType(32, MIRBuilder);
  SPIRVType *ArrayType = nullptr;
  if (NumElems != 0) {
    Register NumElementsVReg =
        buildConstantInt(NumElems, MIRBuilder, SpvTypeInt32, EmitIR);
    ArrayType = createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
      return MIRBuilder.buildInstr(SPIRV::OpTypeArray)
          .addDef(createTypeVReg(MIRBuilder))
          .addUse(getSPIRVTypeID(ElemType))
          .addUse(NumElementsVReg);
    });
  } else {
    ArrayType = createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
      return MIRBuilder.buildInstr(SPIRV::OpTypeRuntimeArray)
          .addDef(createTypeVReg(MIRBuilder))
          .addUse(getSPIRVTypeID(ElemType));
    });
  }

  if (ExplicitLayoutRequired && !isResourceType(ElemType)) {
    Type *ET = const_cast<Type *>(getTypeForSPIRVType(ElemType));
    addArrayStrideDecorations(ArrayType->defs().begin()->getReg(), ET,
                              MIRBuilder);
  }

  return ArrayType;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeOpaque(const StructType *Ty,
                                                MachineIRBuilder &MIRBuilder) {
  assert(Ty->hasName());
  const StringRef Name = Ty->hasName() ? Ty->getName() : "";
  Register ResVReg = createTypeVReg(MIRBuilder);
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeOpaque).addDef(ResVReg);
    addStringImm(Name, MIB);
    buildOpName(ResVReg, Name, MIRBuilder);
    return MIB;
  });
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeStruct(
    const StructType *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual,
    bool ExplicitLayoutRequired, bool EmitIR) {
  const SPIRVSubtarget &ST =
      cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
  SmallVector<Register, 4> FieldTypes;
  constexpr unsigned MaxWordCount = UINT16_MAX;
  const size_t NumElements = Ty->getNumElements();

  size_t MaxNumElements = MaxWordCount - 2;
  size_t SPIRVStructNumElements = NumElements;
  if (NumElements > MaxNumElements) {
    // Do adjustments for continued instructions.
    SPIRVStructNumElements = MaxNumElements;
    MaxNumElements = MaxWordCount - 1;
  }

  for (const auto &Elem : Ty->elements()) {
    SPIRVType *ElemTy = findSPIRVType(toTypedPointer(Elem), MIRBuilder, AccQual,
                                      ExplicitLayoutRequired, EmitIR);
    assert(ElemTy && ElemTy->getOpcode() != SPIRV::OpTypeVoid &&
           "Invalid struct element type");
    FieldTypes.push_back(getSPIRVTypeID(ElemTy));
  }
  Register ResVReg = createTypeVReg(MIRBuilder);
  if (Ty->hasName())
    buildOpName(ResVReg, Ty->getName(), MIRBuilder);
  if (Ty->isPacked() && !ST.isVulkanEnv())
    buildOpDecorate(ResVReg, MIRBuilder, SPIRV::Decoration::CPacked, {});

  SPIRVType *SPVType =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        auto MIBStruct =
            MIRBuilder.buildInstr(SPIRV::OpTypeStruct).addDef(ResVReg);
        for (size_t I = 0; I < SPIRVStructNumElements; ++I)
          MIBStruct.addUse(FieldTypes[I]);
        for (size_t I = SPIRVStructNumElements; I < NumElements;
             I += MaxNumElements) {
          auto MIBCont =
              MIRBuilder.buildInstr(SPIRV::OpTypeStructContinuedINTEL);
          for (size_t J = I; J < std::min(I + MaxNumElements, NumElements); ++J)
            MIBCont.addUse(FieldTypes[I]);
        }
        return MIBStruct;
      });

  if (ExplicitLayoutRequired)
    addStructOffsetDecorations(SPVType->defs().begin()->getReg(),
                               const_cast<StructType *>(Ty), MIRBuilder);

  return SPVType;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSpecialType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual) {
  assert(isSpecialOpaqueType(Ty) && "Not a special opaque builtin type");
  return SPIRV::lowerBuiltinType(Ty, AccQual, MIRBuilder, this);
}

SPIRVType *SPIRVGlobalRegistry::getOpTypePointer(
    SPIRV::StorageClass::StorageClass SC, SPIRVType *ElemType,
    MachineIRBuilder &MIRBuilder, Register Reg) {
  if (!Reg.isValid())
    Reg = createTypeVReg(MIRBuilder);

  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypePointer)
        .addDef(Reg)
        .addImm(static_cast<uint32_t>(SC))
        .addUse(getSPIRVTypeID(ElemType));
  });
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeForwardPointer(
    SPIRV::StorageClass::StorageClass SC, MachineIRBuilder &MIRBuilder) {
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    return MIRBuilder.buildInstr(SPIRV::OpTypeForwardPointer)
        .addUse(createTypeVReg(MIRBuilder))
        .addImm(static_cast<uint32_t>(SC));
  });
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeFunction(
    SPIRVType *RetType, const SmallVectorImpl<SPIRVType *> &ArgTypes,
    MachineIRBuilder &MIRBuilder) {
  return createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeFunction)
                   .addDef(createTypeVReg(MIRBuilder))
                   .addUse(getSPIRVTypeID(RetType));
    for (const SPIRVType *ArgType : ArgTypes)
      MIB.addUse(getSPIRVTypeID(ArgType));
    return MIB;
  });
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeFunctionWithArgs(
    const Type *Ty, SPIRVType *RetType,
    const SmallVectorImpl<SPIRVType *> &ArgTypes,
    MachineIRBuilder &MIRBuilder) {
  if (const MachineInstr *MI = findMI(Ty, false, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI = getOpTypeFunction(RetType, ArgTypes, MIRBuilder);
  add(Ty, false, NewMI);
  return finishCreatingSPIRVType(Ty, NewMI);
}

SPIRVType *SPIRVGlobalRegistry::findSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual,
    bool ExplicitLayoutRequired, bool EmitIR) {
  Ty = adjustIntTypeByWidth(Ty);
  // TODO: findMI needs to know if a layout is required.
  if (const MachineInstr *MI =
          findMI(Ty, ExplicitLayoutRequired, &MIRBuilder.getMF()))
    return MI;
  if (auto It = ForwardPointerTypes.find(Ty); It != ForwardPointerTypes.end())
    return It->second;
  return restOfCreateSPIRVType(Ty, MIRBuilder, AccQual, ExplicitLayoutRequired,
                               EmitIR);
}

Register SPIRVGlobalRegistry::getSPIRVTypeID(const SPIRVType *SpirvType) const {
  assert(SpirvType && "Attempting to get type id for nullptr type.");
  if (SpirvType->getOpcode() == SPIRV::OpTypeForwardPointer ||
      SpirvType->getOpcode() == SPIRV::OpTypeStructContinuedINTEL)
    return SpirvType->uses().begin()->getReg();
  return SpirvType->defs().begin()->getReg();
}

// We need to use a new LLVM integer type if there is a mismatch between
// number of bits in LLVM and SPIRV integer types to let DuplicateTracker
// ensure uniqueness of a SPIRV type by the corresponding LLVM type. Without
// such an adjustment SPIRVGlobalRegistry::getOpTypeInt() could create the
// same "OpTypeInt 8" type for a series of LLVM integer types with number of
// bits less than 8. This would lead to duplicate type definitions
// eventually due to the method that DuplicateTracker utilizes to reason
// about uniqueness of type records.
const Type *SPIRVGlobalRegistry::adjustIntTypeByWidth(const Type *Ty) const {
  if (auto IType = dyn_cast<IntegerType>(Ty)) {
    unsigned SrcBitWidth = IType->getBitWidth();
    if (SrcBitWidth > 1) {
      unsigned BitWidth = adjustOpTypeIntWidth(SrcBitWidth);
      // Maybe change source LLVM type to keep DuplicateTracker consistent.
      if (SrcBitWidth != BitWidth)
        Ty = IntegerType::get(Ty->getContext(), BitWidth);
    }
  }
  return Ty;
}

SPIRVType *SPIRVGlobalRegistry::createSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual,
    bool ExplicitLayoutRequired, bool EmitIR) {
  if (isSpecialOpaqueType(Ty))
    return getOrCreateSpecialType(Ty, MIRBuilder, AccQual);

  if (const MachineInstr *MI =
          findMI(Ty, ExplicitLayoutRequired, &MIRBuilder.getMF()))
    return MI;

  if (auto IType = dyn_cast<IntegerType>(Ty)) {
    const unsigned Width = IType->getBitWidth();
    return Width == 1 ? getOpTypeBool(MIRBuilder)
                      : getOpTypeInt(Width, MIRBuilder, false);
  }
  if (Ty->isFloatingPointTy())
    return getOpTypeFloat(Ty->getPrimitiveSizeInBits(), MIRBuilder);
  if (Ty->isVoidTy())
    return getOpTypeVoid(MIRBuilder);
  if (Ty->isVectorTy()) {
    SPIRVType *El =
        findSPIRVType(cast<FixedVectorType>(Ty)->getElementType(), MIRBuilder,
                      AccQual, ExplicitLayoutRequired, EmitIR);
    return getOpTypeVector(cast<FixedVectorType>(Ty)->getNumElements(), El,
                           MIRBuilder);
  }
  if (Ty->isArrayTy()) {
    SPIRVType *El = findSPIRVType(Ty->getArrayElementType(), MIRBuilder,
                                  AccQual, ExplicitLayoutRequired, EmitIR);
    return getOpTypeArray(Ty->getArrayNumElements(), El, MIRBuilder,
                          ExplicitLayoutRequired, EmitIR);
  }
  if (auto SType = dyn_cast<StructType>(Ty)) {
    if (SType->isOpaque())
      return getOpTypeOpaque(SType, MIRBuilder);
    return getOpTypeStruct(SType, MIRBuilder, AccQual, ExplicitLayoutRequired,
                           EmitIR);
  }
  if (auto FType = dyn_cast<FunctionType>(Ty)) {
    SPIRVType *RetTy = findSPIRVType(FType->getReturnType(), MIRBuilder,
                                     AccQual, ExplicitLayoutRequired, EmitIR);
    SmallVector<SPIRVType *, 4> ParamTypes;
    for (const auto &ParamTy : FType->params())
      ParamTypes.push_back(findSPIRVType(ParamTy, MIRBuilder, AccQual,
                                         ExplicitLayoutRequired, EmitIR));
    return getOpTypeFunction(RetTy, ParamTypes, MIRBuilder);
  }

  unsigned AddrSpace = typeToAddressSpace(Ty);
  SPIRVType *SpvElementType = nullptr;
  if (Type *ElemTy = ::getPointeeType(Ty))
    SpvElementType = getOrCreateSPIRVType(ElemTy, MIRBuilder, AccQual, EmitIR);
  else
    SpvElementType = getOrCreateSPIRVIntegerType(8, MIRBuilder);

  // Get access to information about available extensions
  const SPIRVSubtarget *ST =
      static_cast<const SPIRVSubtarget *>(&MIRBuilder.getMF().getSubtarget());
  auto SC = addressSpaceToStorageClass(AddrSpace, *ST);

  Type *ElemTy = ::getPointeeType(Ty);
  if (!ElemTy) {
    ElemTy = Type::getInt8Ty(MIRBuilder.getContext());
  }

  // If we have forward pointer associated with this type, use its register
  // operand to create OpTypePointer.
  if (auto It = ForwardPointerTypes.find(Ty); It != ForwardPointerTypes.end()) {
    Register Reg = getSPIRVTypeID(It->second);
    // TODO: what does getOpTypePointer do?
    return getOpTypePointer(SC, SpvElementType, MIRBuilder, Reg);
  }

  return getOrCreateSPIRVPointerType(ElemTy, MIRBuilder, SC);
}

SPIRVType *SPIRVGlobalRegistry::restOfCreateSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual,
    bool ExplicitLayoutRequired, bool EmitIR) {
  // TODO: Could this create a problem if one requires an explicit layout, and
  // the next time it does not?
  if (TypesInProcessing.count(Ty) && !isPointerTyOrWrapper(Ty))
    return nullptr;
  TypesInProcessing.insert(Ty);
  SPIRVType *SpirvType = createSPIRVType(Ty, MIRBuilder, AccessQual,
                                         ExplicitLayoutRequired, EmitIR);
  TypesInProcessing.erase(Ty);
  VRegToTypeMap[&MIRBuilder.getMF()][getSPIRVTypeID(SpirvType)] = SpirvType;

  // TODO: We could end up with two SPIR-V types pointing to the same llvm type.
  // Is that a problem?
  SPIRVToLLVMType[SpirvType] = unifyPtrType(Ty);

  if (SpirvType->getOpcode() == SPIRV::OpTypeForwardPointer ||
      findMI(Ty, false, &MIRBuilder.getMF()) || isSpecialOpaqueType(Ty))
    return SpirvType;

  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty);
      ExtTy && isTypedPointerWrapper(ExtTy))
    add(ExtTy->getTypeParameter(0), ExtTy->getIntParameter(0), SpirvType);
  else if (!isPointerTy(Ty))
    add(Ty, ExplicitLayoutRequired, SpirvType);
  else if (isTypedPointerTy(Ty))
    add(cast<TypedPointerType>(Ty)->getElementType(),
        getPointerAddressSpace(Ty), SpirvType);
  else
    add(Type::getInt8Ty(MIRBuilder.getMF().getFunction().getContext()),
        getPointerAddressSpace(Ty), SpirvType);
  return SpirvType;
}

SPIRVType *
SPIRVGlobalRegistry::getSPIRVTypeForVReg(Register VReg,
                                         const MachineFunction *MF) const {
  auto t = VRegToTypeMap.find(MF ? MF : CurMF);
  if (t != VRegToTypeMap.end()) {
    auto tt = t->second.find(VReg);
    if (tt != t->second.end())
      return tt->second;
  }
  return nullptr;
}

SPIRVType *SPIRVGlobalRegistry::getResultType(Register VReg,
                                              MachineFunction *MF) {
  if (!MF)
    MF = CurMF;
  MachineInstr *Instr = getVRegDef(MF->getRegInfo(), VReg);
  return getSPIRVTypeForVReg(Instr->getOperand(1).getReg(), MF);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual,
    bool ExplicitLayoutRequired, bool EmitIR) {
  const MachineFunction *MF = &MIRBuilder.getMF();
  Register Reg;
  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty);
      ExtTy && isTypedPointerWrapper(ExtTy))
    Reg = find(ExtTy->getTypeParameter(0), ExtTy->getIntParameter(0), MF);
  else if (!isPointerTy(Ty))
    Reg = find(Ty = adjustIntTypeByWidth(Ty), ExplicitLayoutRequired, MF);
  else if (isTypedPointerTy(Ty))
    Reg = find(cast<TypedPointerType>(Ty)->getElementType(),
               getPointerAddressSpace(Ty), MF);
  else
    Reg = find(Type::getInt8Ty(MIRBuilder.getMF().getFunction().getContext()),
               getPointerAddressSpace(Ty), MF);
  if (Reg.isValid() && !isSpecialOpaqueType(Ty))
    return getSPIRVTypeForVReg(Reg);

  TypesInProcessing.clear();
  SPIRVType *STy = restOfCreateSPIRVType(Ty, MIRBuilder, AccessQual,
                                         ExplicitLayoutRequired, EmitIR);
  // Create normal pointer types for the corresponding OpTypeForwardPointers.
  for (auto &CU : ForwardPointerTypes) {
    // Pointer type themselves do not require an explicit layout. The types
    // they pointer to might, but that is taken care of when creating the type.
    bool PtrNeedsLayout = false;
    const Type *Ty2 = CU.first;
    SPIRVType *STy2 = CU.second;
    if ((Reg = find(Ty2, PtrNeedsLayout, MF)).isValid())
      STy2 = getSPIRVTypeForVReg(Reg);
    else
      STy2 = restOfCreateSPIRVType(Ty2, MIRBuilder, AccessQual, PtrNeedsLayout,
                                   EmitIR);
    if (Ty == Ty2)
      STy = STy2;
  }
  ForwardPointerTypes.clear();
  return STy;
}

bool SPIRVGlobalRegistry::isScalarOfType(Register VReg,
                                         unsigned TypeOpcode) const {
  SPIRVType *Type = getSPIRVTypeForVReg(VReg);
  assert(Type && "isScalarOfType VReg has no type assigned");
  return Type->getOpcode() == TypeOpcode;
}

bool SPIRVGlobalRegistry::isScalarOrVectorOfType(Register VReg,
                                                 unsigned TypeOpcode) const {
  SPIRVType *Type = getSPIRVTypeForVReg(VReg);
  assert(Type && "isScalarOrVectorOfType VReg has no type assigned");
  if (Type->getOpcode() == TypeOpcode)
    return true;
  if (Type->getOpcode() == SPIRV::OpTypeVector) {
    Register ScalarTypeVReg = Type->getOperand(1).getReg();
    SPIRVType *ScalarType = getSPIRVTypeForVReg(ScalarTypeVReg);
    return ScalarType->getOpcode() == TypeOpcode;
  }
  return false;
}

bool SPIRVGlobalRegistry::isResourceType(SPIRVType *Type) const {
  switch (Type->getOpcode()) {
  case SPIRV::OpTypeImage:
  case SPIRV::OpTypeSampler:
  case SPIRV::OpTypeSampledImage:
    return true;
  case SPIRV::OpTypeStruct:
    return hasBlockDecoration(Type);
  default:
    return false;
  }
  return false;
}
unsigned
SPIRVGlobalRegistry::getScalarOrVectorComponentCount(Register VReg) const {
  return getScalarOrVectorComponentCount(getSPIRVTypeForVReg(VReg));
}

unsigned
SPIRVGlobalRegistry::getScalarOrVectorComponentCount(SPIRVType *Type) const {
  if (!Type)
    return 0;
  return Type->getOpcode() == SPIRV::OpTypeVector
             ? static_cast<unsigned>(Type->getOperand(2).getImm())
             : 1;
}

SPIRVType *
SPIRVGlobalRegistry::getScalarOrVectorComponentType(Register VReg) const {
  return getScalarOrVectorComponentType(getSPIRVTypeForVReg(VReg));
}

SPIRVType *
SPIRVGlobalRegistry::getScalarOrVectorComponentType(SPIRVType *Type) const {
  if (!Type)
    return nullptr;
  Register ScalarReg = Type->getOpcode() == SPIRV::OpTypeVector
                           ? Type->getOperand(1).getReg()
                           : Type->getOperand(0).getReg();
  SPIRVType *ScalarType = getSPIRVTypeForVReg(ScalarReg);
  assert(isScalarOrVectorOfType(Type->getOperand(0).getReg(),
                                ScalarType->getOpcode()));
  return ScalarType;
}

unsigned
SPIRVGlobalRegistry::getScalarOrVectorBitWidth(const SPIRVType *Type) const {
  assert(Type && "Invalid Type pointer");
  if (Type->getOpcode() == SPIRV::OpTypeVector) {
    auto EleTypeReg = Type->getOperand(1).getReg();
    Type = getSPIRVTypeForVReg(EleTypeReg);
  }
  if (Type->getOpcode() == SPIRV::OpTypeInt ||
      Type->getOpcode() == SPIRV::OpTypeFloat)
    return Type->getOperand(1).getImm();
  if (Type->getOpcode() == SPIRV::OpTypeBool)
    return 1;
  llvm_unreachable("Attempting to get bit width of non-integer/float type.");
}

unsigned SPIRVGlobalRegistry::getNumScalarOrVectorTotalBitWidth(
    const SPIRVType *Type) const {
  assert(Type && "Invalid Type pointer");
  unsigned NumElements = 1;
  if (Type->getOpcode() == SPIRV::OpTypeVector) {
    NumElements = static_cast<unsigned>(Type->getOperand(2).getImm());
    Type = getSPIRVTypeForVReg(Type->getOperand(1).getReg());
  }
  return Type->getOpcode() == SPIRV::OpTypeInt ||
                 Type->getOpcode() == SPIRV::OpTypeFloat
             ? NumElements * Type->getOperand(1).getImm()
             : 0;
}

const SPIRVType *SPIRVGlobalRegistry::retrieveScalarOrVectorIntType(
    const SPIRVType *Type) const {
  if (Type && Type->getOpcode() == SPIRV::OpTypeVector)
    Type = getSPIRVTypeForVReg(Type->getOperand(1).getReg());
  return Type && Type->getOpcode() == SPIRV::OpTypeInt ? Type : nullptr;
}

bool SPIRVGlobalRegistry::isScalarOrVectorSigned(const SPIRVType *Type) const {
  const SPIRVType *IntType = retrieveScalarOrVectorIntType(Type);
  return IntType && IntType->getOperand(2).getImm() != 0;
}

SPIRVType *SPIRVGlobalRegistry::getPointeeType(SPIRVType *PtrType) {
  return PtrType && PtrType->getOpcode() == SPIRV::OpTypePointer
             ? getSPIRVTypeForVReg(PtrType->getOperand(2).getReg())
             : nullptr;
}

unsigned SPIRVGlobalRegistry::getPointeeTypeOp(Register PtrReg) {
  SPIRVType *ElemType = getPointeeType(getSPIRVTypeForVReg(PtrReg));
  return ElemType ? ElemType->getOpcode() : 0;
}

bool SPIRVGlobalRegistry::isBitcastCompatible(const SPIRVType *Type1,
                                              const SPIRVType *Type2) const {
  if (!Type1 || !Type2)
    return false;
  auto Op1 = Type1->getOpcode(), Op2 = Type2->getOpcode();
  // Ignore difference between <1.5 and >=1.5 protocol versions:
  // it's valid if either Result Type or Operand is a pointer, and the other
  // is a pointer, an integer scalar, or an integer vector.
  if (Op1 == SPIRV::OpTypePointer &&
      (Op2 == SPIRV::OpTypePointer || retrieveScalarOrVectorIntType(Type2)))
    return true;
  if (Op2 == SPIRV::OpTypePointer &&
      (Op1 == SPIRV::OpTypePointer || retrieveScalarOrVectorIntType(Type1)))
    return true;
  unsigned Bits1 = getNumScalarOrVectorTotalBitWidth(Type1),
           Bits2 = getNumScalarOrVectorTotalBitWidth(Type2);
  return Bits1 > 0 && Bits1 == Bits2;
}

SPIRV::StorageClass::StorageClass
SPIRVGlobalRegistry::getPointerStorageClass(Register VReg) const {
  SPIRVType *Type = getSPIRVTypeForVReg(VReg);
  assert(Type && Type->getOpcode() == SPIRV::OpTypePointer &&
         Type->getOperand(1).isImm() && "Pointer type is expected");
  return getPointerStorageClass(Type);
}

SPIRV::StorageClass::StorageClass
SPIRVGlobalRegistry::getPointerStorageClass(const SPIRVType *Type) const {
  return static_cast<SPIRV::StorageClass::StorageClass>(
      Type->getOperand(1).getImm());
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateVulkanBufferType(
    MachineIRBuilder &MIRBuilder, Type *ElemType,
    SPIRV::StorageClass::StorageClass SC, bool IsWritable, bool EmitIr) {
  auto Key = SPIRV::irhandle_vkbuffer(ElemType, SC, IsWritable);
  if (const MachineInstr *MI = findMI(Key, &MIRBuilder.getMF()))
    return MI;

  bool ExplicitLayoutRequired = storageClassRequiresExplictLayout(SC);
  // We need to get the SPIR-V type for the element here, so we can add the
  // decoration to it.
  auto *T = StructType::create(ElemType);
  auto *BlockType =
      getOrCreateSPIRVType(T, MIRBuilder, SPIRV::AccessQualifier::None,
                           ExplicitLayoutRequired, EmitIr);

  buildOpDecorate(BlockType->defs().begin()->getReg(), MIRBuilder,
                  SPIRV::Decoration::Block, {});

  if (!IsWritable) {
    buildOpMemberDecorate(BlockType->defs().begin()->getReg(), MIRBuilder,
                          SPIRV::Decoration::NonWritable, 0, {});
  }

  SPIRVType *R = getOrCreateSPIRVPointerTypeInternal(BlockType, MIRBuilder, SC);
  add(Key, R);
  return R;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeImage(
    MachineIRBuilder &MIRBuilder, SPIRVType *SampledType, SPIRV::Dim::Dim Dim,
    uint32_t Depth, uint32_t Arrayed, uint32_t Multisampled, uint32_t Sampled,
    SPIRV::ImageFormat::ImageFormat ImageFormat,
    SPIRV::AccessQualifier::AccessQualifier AccessQual) {
  auto Key = SPIRV::irhandle_image(SPIRVToLLVMType.lookup(SampledType), Dim,
                                   Depth, Arrayed, Multisampled, Sampled,
                                   ImageFormat, AccessQual);
  if (const MachineInstr *MI = findMI(Key, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        auto MIB =
            MIRBuilder.buildInstr(SPIRV::OpTypeImage)
                .addDef(createTypeVReg(MIRBuilder))
                .addUse(getSPIRVTypeID(SampledType))
                .addImm(Dim)
                .addImm(Depth)   // Depth (whether or not it is a Depth image).
                .addImm(Arrayed) // Arrayed.
                .addImm(Multisampled) // Multisampled (0 = only single-sample).
                .addImm(Sampled)      // Sampled (0 = usage known at runtime).
                .addImm(ImageFormat);
        if (AccessQual != SPIRV::AccessQualifier::None)
          MIB.addImm(AccessQual);
        return MIB;
      });
  add(Key, NewMI);
  return NewMI;
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateOpTypeSampler(MachineIRBuilder &MIRBuilder) {
  auto Key = SPIRV::irhandle_sampler();
  const MachineFunction *MF = &MIRBuilder.getMF();
  if (const MachineInstr *MI = findMI(Key, MF))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(SPIRV::OpTypeSampler)
            .addDef(createTypeVReg(MIRBuilder));
      });
  add(Key, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypePipe(
    MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual) {
  auto Key = SPIRV::irhandle_pipe(AccessQual);
  if (const MachineInstr *MI = findMI(Key, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(SPIRV::OpTypePipe)
            .addDef(createTypeVReg(MIRBuilder))
            .addImm(AccessQual);
      });
  add(Key, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeDeviceEvent(
    MachineIRBuilder &MIRBuilder) {
  auto Key = SPIRV::irhandle_event();
  if (const MachineInstr *MI = findMI(Key, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(SPIRV::OpTypeDeviceEvent)
            .addDef(createTypeVReg(MIRBuilder));
      });
  add(Key, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeSampledImage(
    SPIRVType *ImageType, MachineIRBuilder &MIRBuilder) {
  auto Key = SPIRV::irhandle_sampled_image(
      SPIRVToLLVMType.lookup(MIRBuilder.getMF().getRegInfo().getVRegDef(
          ImageType->getOperand(1).getReg())),
      ImageType);
  if (const MachineInstr *MI = findMI(Key, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(SPIRV::OpTypeSampledImage)
            .addDef(createTypeVReg(MIRBuilder))
            .addUse(getSPIRVTypeID(ImageType));
      });
  add(Key, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeCoopMatr(
    MachineIRBuilder &MIRBuilder, const TargetExtType *ExtensionType,
    const SPIRVType *ElemType, uint32_t Scope, uint32_t Rows, uint32_t Columns,
    uint32_t Use, bool EmitIR) {
  if (const MachineInstr *MI =
          findMI(ExtensionType, false, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        SPIRVType *SpvTypeInt32 = getOrCreateSPIRVIntegerType(32, MIRBuilder);
        const Type *ET = getTypeForSPIRVType(ElemType);
        if (ET->isIntegerTy() && ET->getIntegerBitWidth() == 4 &&
            cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget())
                .canUseExtension(SPIRV::Extension::SPV_INTEL_int4)) {
          MIRBuilder.buildInstr(SPIRV::OpCapability)
              .addImm(SPIRV::Capability::Int4CooperativeMatrixINTEL);
        }
        return MIRBuilder.buildInstr(SPIRV::OpTypeCooperativeMatrixKHR)
            .addDef(createTypeVReg(MIRBuilder))
            .addUse(getSPIRVTypeID(ElemType))
            .addUse(buildConstantInt(Scope, MIRBuilder, SpvTypeInt32, EmitIR))
            .addUse(buildConstantInt(Rows, MIRBuilder, SpvTypeInt32, EmitIR))
            .addUse(buildConstantInt(Columns, MIRBuilder, SpvTypeInt32, EmitIR))
            .addUse(buildConstantInt(Use, MIRBuilder, SpvTypeInt32, EmitIR));
      });
  add(ExtensionType, false, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeByOpcode(
    const Type *Ty, MachineIRBuilder &MIRBuilder, unsigned Opcode) {
  if (const MachineInstr *MI = findMI(Ty, false, &MIRBuilder.getMF()))
    return MI;
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return MIRBuilder.buildInstr(Opcode).addDef(createTypeVReg(MIRBuilder));
      });
  add(Ty, false, NewMI);
  return NewMI;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateUnknownType(
    const Type *Ty, MachineIRBuilder &MIRBuilder, unsigned Opcode,
    const ArrayRef<MCOperand> Operands) {
  if (const MachineInstr *MI = findMI(Ty, false, &MIRBuilder.getMF()))
    return MI;
  Register ResVReg = createTypeVReg(MIRBuilder);
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        MachineInstrBuilder MIB = MIRBuilder.buildInstr(SPIRV::UNKNOWN_type)
                                      .addDef(ResVReg)
                                      .addImm(Opcode);
        for (MCOperand Operand : Operands) {
          if (Operand.isReg()) {
            MIB.addUse(Operand.getReg());
          } else if (Operand.isImm()) {
            MIB.addImm(Operand.getImm());
          }
        }
        return MIB;
      });
  add(Ty, false, NewMI);
  return NewMI;
}

// Returns nullptr if unable to recognize SPIRV type name
SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVTypeByName(
    StringRef TypeStr, MachineIRBuilder &MIRBuilder, bool EmitIR,
    SPIRV::StorageClass::StorageClass SC,
    SPIRV::AccessQualifier::AccessQualifier AQ) {
  unsigned VecElts = 0;
  auto &Ctx = MIRBuilder.getMF().getFunction().getContext();

  // Parse strings representing either a SPIR-V or OpenCL builtin type.
  if (hasBuiltinTypePrefix(TypeStr))
    return getOrCreateSPIRVType(SPIRV::parseBuiltinTypeNameToTargetExtType(
                                    TypeStr.str(), MIRBuilder.getContext()),
                                MIRBuilder, AQ, false, true);

  // Parse type name in either "typeN" or "type vector[N]" format, where
  // N is the number of elements of the vector.
  Type *Ty;

  Ty = parseBasicTypeName(TypeStr, Ctx);
  if (!Ty)
    // Unable to recognize SPIRV type name
    return nullptr;

  const SPIRVType *SpirvTy =
      getOrCreateSPIRVType(Ty, MIRBuilder, AQ, false, true);

  // Handle "type*" or  "type* vector[N]".
  if (TypeStr.consume_front("*"))
    SpirvTy = getOrCreateSPIRVPointerType(Ty, MIRBuilder, SC);

  // Handle "typeN*" or  "type vector[N]*".
  bool IsPtrToVec = TypeStr.consume_back("*");

  if (TypeStr.consume_front(" vector[")) {
    TypeStr = TypeStr.substr(0, TypeStr.find(']'));
  }
  TypeStr.getAsInteger(10, VecElts);
  if (VecElts > 0)
    SpirvTy = getOrCreateSPIRVVectorType(SpirvTy, VecElts, MIRBuilder, EmitIR);

  if (IsPtrToVec)
    SpirvTy = getOrCreateSPIRVPointerType(SpirvTy, MIRBuilder, SC);

  return SpirvTy;
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                                 MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), BitWidth),
      MIRBuilder, SPIRV::AccessQualifier::ReadWrite, false, true);
}

SPIRVType *SPIRVGlobalRegistry::finishCreatingSPIRVType(const Type *LLVMTy,
                                                        SPIRVType *SpirvType) {
  assert(CurMF == SpirvType->getMF());
  VRegToTypeMap[CurMF][getSPIRVTypeID(SpirvType)] = SpirvType;
  SPIRVToLLVMType[SpirvType] = unifyPtrType(LLVMTy);
  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVType(unsigned BitWidth,
                                                     MachineInstr &I,
                                                     const SPIRVInstrInfo &TII,
                                                     unsigned SPIRVOPcode,
                                                     Type *Ty) {
  if (const MachineInstr *MI = findMI(Ty, false, CurMF))
    return MI;
  MachineBasicBlock &DepMBB = I.getMF()->front();
  MachineIRBuilder MIRBuilder(DepMBB, DepMBB.getFirstNonPHI());
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return BuildMI(MIRBuilder.getMBB(), *MIRBuilder.getInsertPt(),
                       MIRBuilder.getDL(), TII.get(SPIRVOPcode))
            .addDef(createTypeVReg(CurMF->getRegInfo()))
            .addImm(BitWidth)
            .addImm(0);
      });
  add(Ty, false, NewMI);
  return finishCreatingSPIRVType(Ty, NewMI);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(
    unsigned BitWidth, MachineInstr &I, const SPIRVInstrInfo &TII) {
  // Maybe adjust bit width to keep DuplicateTracker consistent. Without
  // such an adjustment SPIRVGlobalRegistry::getOpTypeInt() could create, for
  // example, the same "OpTypeInt 8" type for a series of LLVM integer types
  // with number of bits less than 8, causing duplicate type definitions.
  if (BitWidth > 1)
    BitWidth = adjustOpTypeIntWidth(BitWidth);
  Type *LLVMTy = IntegerType::get(CurMF->getFunction().getContext(), BitWidth);
  return getOrCreateSPIRVType(BitWidth, I, TII, SPIRV::OpTypeInt, LLVMTy);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVFloatType(
    unsigned BitWidth, MachineInstr &I, const SPIRVInstrInfo &TII) {
  LLVMContext &Ctx = CurMF->getFunction().getContext();
  Type *LLVMTy;
  switch (BitWidth) {
  case 16:
    LLVMTy = Type::getHalfTy(Ctx);
    break;
  case 32:
    LLVMTy = Type::getFloatTy(Ctx);
    break;
  case 64:
    LLVMTy = Type::getDoubleTy(Ctx);
    break;
  default:
    llvm_unreachable("Bit width is of unexpected size.");
  }
  return getOrCreateSPIRVType(BitWidth, I, TII, SPIRV::OpTypeFloat, LLVMTy);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder,
                                              bool EmitIR) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), 1),
      MIRBuilder, SPIRV::AccessQualifier::ReadWrite, false, EmitIR);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVBoolType(MachineInstr &I,
                                              const SPIRVInstrInfo &TII) {
  Type *Ty = IntegerType::get(CurMF->getFunction().getContext(), 1);
  if (const MachineInstr *MI = findMI(Ty, false, CurMF))
    return MI;
  MachineBasicBlock &DepMBB = I.getMF()->front();
  MachineIRBuilder MIRBuilder(DepMBB, DepMBB.getFirstNonPHI());
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return BuildMI(MIRBuilder.getMBB(), *MIRBuilder.getInsertPt(),
                       MIRBuilder.getDL(), TII.get(SPIRV::OpTypeBool))
            .addDef(createTypeVReg(CurMF->getRegInfo()));
      });
  add(Ty, false, NewMI);
  return finishCreatingSPIRVType(Ty, NewMI);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVVectorType(
    SPIRVType *BaseType, unsigned NumElements, MachineIRBuilder &MIRBuilder,
    bool EmitIR) {
  return getOrCreateSPIRVType(
      FixedVectorType::get(const_cast<Type *>(getTypeForSPIRVType(BaseType)),
                           NumElements),
      MIRBuilder, SPIRV::AccessQualifier::ReadWrite, false, EmitIR);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVVectorType(
    SPIRVType *BaseType, unsigned NumElements, MachineInstr &I,
    const SPIRVInstrInfo &TII) {
  Type *Ty = FixedVectorType::get(
      const_cast<Type *>(getTypeForSPIRVType(BaseType)), NumElements);
  if (const MachineInstr *MI = findMI(Ty, false, CurMF))
    return MI;
  MachineInstr *DepMI = const_cast<MachineInstr *>(BaseType);
  MachineIRBuilder MIRBuilder(*DepMI->getParent(), DepMI->getIterator());
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return BuildMI(MIRBuilder.getMBB(), *MIRBuilder.getInsertPt(),
                       MIRBuilder.getDL(), TII.get(SPIRV::OpTypeVector))
            .addDef(createTypeVReg(CurMF->getRegInfo()))
            .addUse(getSPIRVTypeID(BaseType))
            .addImm(NumElements);
      });
  add(Ty, false, NewMI);
  return finishCreatingSPIRVType(Ty, NewMI);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    const Type *BaseType, MachineInstr &I,
    SPIRV::StorageClass::StorageClass SC) {
  MachineIRBuilder MIRBuilder(I);
  return getOrCreateSPIRVPointerType(BaseType, MIRBuilder, SC);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    const Type *BaseType, MachineIRBuilder &MIRBuilder,
    SPIRV::StorageClass::StorageClass SC) {
  // TODO: Need to check if EmitIr should always be true.
  SPIRVType *SpirvBaseType = getOrCreateSPIRVType(
      BaseType, MIRBuilder, SPIRV::AccessQualifier::ReadWrite,
      storageClassRequiresExplictLayout(SC), true);
  assert(SpirvBaseType);
  return getOrCreateSPIRVPointerTypeInternal(SpirvBaseType, MIRBuilder, SC);
}

SPIRVType *SPIRVGlobalRegistry::changePointerStorageClass(
    SPIRVType *PtrType, SPIRV::StorageClass::StorageClass SC, MachineInstr &I) {
  [[maybe_unused]] SPIRV::StorageClass::StorageClass OldSC =
      getPointerStorageClass(PtrType);
  assert(storageClassRequiresExplictLayout(OldSC) ==
         storageClassRequiresExplictLayout(SC));

  SPIRVType *PointeeType = getPointeeType(PtrType);
  MachineIRBuilder MIRBuilder(I);
  return getOrCreateSPIRVPointerTypeInternal(PointeeType, MIRBuilder, SC);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
    SPIRV::StorageClass::StorageClass SC) {
  const Type *LLVMType = getTypeForSPIRVType(BaseType);
  assert(!storageClassRequiresExplictLayout(SC));
  SPIRVType *R = getOrCreateSPIRVPointerType(LLVMType, MIRBuilder, SC);
  assert(
      getPointeeType(R) == BaseType &&
      "The base type was not correctly laid out for the given storage class.");
  return R;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerTypeInternal(
    SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
    SPIRV::StorageClass::StorageClass SC) {
  const Type *PointerElementType = getTypeForSPIRVType(BaseType);
  unsigned AddressSpace = storageClassToAddressSpace(SC);
  if (const MachineInstr *MI = findMI(PointerElementType, AddressSpace, CurMF))
    return MI;
  Type *Ty = TypedPointerType::get(const_cast<Type *>(PointerElementType),
                                   AddressSpace);
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        return BuildMI(MIRBuilder.getMBB(), MIRBuilder.getInsertPt(),
                       MIRBuilder.getDebugLoc(),
                       MIRBuilder.getTII().get(SPIRV::OpTypePointer))
            .addDef(createTypeVReg(CurMF->getRegInfo()))
            .addImm(static_cast<uint32_t>(SC))
            .addUse(getSPIRVTypeID(BaseType));
      });
  add(PointerElementType, AddressSpace, NewMI);
  return finishCreatingSPIRVType(Ty, NewMI);
}

Register SPIRVGlobalRegistry::getOrCreateUndef(MachineInstr &I,
                                               SPIRVType *SpvType,
                                               const SPIRVInstrInfo &TII) {
  UndefValue *UV =
      UndefValue::get(const_cast<Type *>(getTypeForSPIRVType(SpvType)));
  Register Res = find(UV, CurMF);
  if (Res.isValid())
    return Res;

  LLT LLTy = LLT::scalar(64);
  Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::iIDRegClass);
  assignSPIRVTypeToVReg(SpvType, Res, *CurMF);

  MachineInstr *DepMI = const_cast<MachineInstr *>(SpvType);
  MachineIRBuilder MIRBuilder(*DepMI->getParent(), DepMI->getIterator());
  const MachineInstr *NewMI =
      createOpType(MIRBuilder, [&](MachineIRBuilder &MIRBuilder) {
        auto MIB = BuildMI(MIRBuilder.getMBB(), *MIRBuilder.getInsertPt(),
                           MIRBuilder.getDL(), TII.get(SPIRV::OpUndef))
                       .addDef(Res)
                       .addUse(getSPIRVTypeID(SpvType));
        const auto &ST = CurMF->getSubtarget();
        constrainSelectedInstRegOperands(*MIB, *ST.getInstrInfo(),
                                         *ST.getRegisterInfo(),
                                         *ST.getRegBankInfo());
        return MIB;
      });
  add(UV, NewMI);
  return Res;
}

const TargetRegisterClass *
SPIRVGlobalRegistry::getRegClass(SPIRVType *SpvType) const {
  unsigned Opcode = SpvType->getOpcode();
  switch (Opcode) {
  case SPIRV::OpTypeFloat:
    return &SPIRV::fIDRegClass;
  case SPIRV::OpTypePointer:
    return &SPIRV::pIDRegClass;
  case SPIRV::OpTypeVector: {
    SPIRVType *ElemType = getSPIRVTypeForVReg(SpvType->getOperand(1).getReg());
    unsigned ElemOpcode = ElemType ? ElemType->getOpcode() : 0;
    if (ElemOpcode == SPIRV::OpTypeFloat)
      return &SPIRV::vfIDRegClass;
    if (ElemOpcode == SPIRV::OpTypePointer)
      return &SPIRV::vpIDRegClass;
    return &SPIRV::vIDRegClass;
  }
  }
  return &SPIRV::iIDRegClass;
}

inline unsigned getAS(SPIRVType *SpvType) {
  return storageClassToAddressSpace(
      static_cast<SPIRV::StorageClass::StorageClass>(
          SpvType->getOperand(1).getImm()));
}

LLT SPIRVGlobalRegistry::getRegType(SPIRVType *SpvType) const {
  unsigned Opcode = SpvType ? SpvType->getOpcode() : 0;
  switch (Opcode) {
  case SPIRV::OpTypeInt:
  case SPIRV::OpTypeFloat:
  case SPIRV::OpTypeBool:
    return LLT::scalar(getScalarOrVectorBitWidth(SpvType));
  case SPIRV::OpTypePointer:
    return LLT::pointer(getAS(SpvType), getPointerSize());
  case SPIRV::OpTypeVector: {
    SPIRVType *ElemType = getSPIRVTypeForVReg(SpvType->getOperand(1).getReg());
    LLT ET;
    switch (ElemType ? ElemType->getOpcode() : 0) {
    case SPIRV::OpTypePointer:
      ET = LLT::pointer(getAS(ElemType), getPointerSize());
      break;
    case SPIRV::OpTypeInt:
    case SPIRV::OpTypeFloat:
    case SPIRV::OpTypeBool:
      ET = LLT::scalar(getScalarOrVectorBitWidth(ElemType));
      break;
    default:
      ET = LLT::scalar(64);
    }
    return LLT::fixed_vector(
        static_cast<unsigned>(SpvType->getOperand(2).getImm()), ET);
  }
  }
  return LLT::scalar(64);
}

// Aliasing list MD contains several scope MD nodes whithin it. Each scope MD
// has a selfreference and an extra MD node for aliasing domain and also it
// can contain an optional string operand. Domain MD contains a self-reference
// with an optional string operand. Here we unfold the list, creating SPIR-V
// aliasing instructions.
// TODO: add support for an optional string operand.
MachineInstr *SPIRVGlobalRegistry::getOrAddMemAliasingINTELInst(
    MachineIRBuilder &MIRBuilder, const MDNode *AliasingListMD) {
  if (AliasingListMD->getNumOperands() == 0)
    return nullptr;
  if (auto L = AliasInstMDMap.find(AliasingListMD); L != AliasInstMDMap.end())
    return L->second;

  SmallVector<MachineInstr *> ScopeList;
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  for (const MDOperand &MDListOp : AliasingListMD->operands()) {
    if (MDNode *ScopeMD = dyn_cast<MDNode>(MDListOp)) {
      if (ScopeMD->getNumOperands() < 2)
        return nullptr;
      MDNode *DomainMD = dyn_cast<MDNode>(ScopeMD->getOperand(1));
      if (!DomainMD)
        return nullptr;
      auto *Domain = [&] {
        auto D = AliasInstMDMap.find(DomainMD);
        if (D != AliasInstMDMap.end())
          return D->second;
        const Register Ret = MRI->createVirtualRegister(&SPIRV::IDRegClass);
        auto MIB =
            MIRBuilder.buildInstr(SPIRV::OpAliasDomainDeclINTEL).addDef(Ret);
        return MIB.getInstr();
      }();
      AliasInstMDMap.insert(std::make_pair(DomainMD, Domain));
      auto *Scope = [&] {
        auto S = AliasInstMDMap.find(ScopeMD);
        if (S != AliasInstMDMap.end())
          return S->second;
        const Register Ret = MRI->createVirtualRegister(&SPIRV::IDRegClass);
        auto MIB = MIRBuilder.buildInstr(SPIRV::OpAliasScopeDeclINTEL)
                       .addDef(Ret)
                       .addUse(Domain->getOperand(0).getReg());
        return MIB.getInstr();
      }();
      AliasInstMDMap.insert(std::make_pair(ScopeMD, Scope));
      ScopeList.push_back(Scope);
    }
  }

  const Register Ret = MRI->createVirtualRegister(&SPIRV::IDRegClass);
  auto MIB =
      MIRBuilder.buildInstr(SPIRV::OpAliasScopeListDeclINTEL).addDef(Ret);
  for (auto *Scope : ScopeList)
    MIB.addUse(Scope->getOperand(0).getReg());
  auto List = MIB.getInstr();
  AliasInstMDMap.insert(std::make_pair(AliasingListMD, List));
  return List;
}

void SPIRVGlobalRegistry::buildMemAliasingOpDecorate(
    Register Reg, MachineIRBuilder &MIRBuilder, uint32_t Dec,
    const MDNode *AliasingListMD) {
  MachineInstr *AliasList =
      getOrAddMemAliasingINTELInst(MIRBuilder, AliasingListMD);
  if (!AliasList)
    return;
  MIRBuilder.buildInstr(SPIRV::OpDecorate)
      .addUse(Reg)
      .addImm(Dec)
      .addUse(AliasList->getOperand(0).getReg());
}
void SPIRVGlobalRegistry::replaceAllUsesWith(Value *Old, Value *New,
                                             bool DeleteOld) {
  Old->replaceAllUsesWith(New);
  updateIfExistDeducedElementType(Old, New, DeleteOld);
  updateIfExistAssignPtrTypeInstr(Old, New, DeleteOld);
}

void SPIRVGlobalRegistry::buildAssignType(IRBuilder<> &B, Type *Ty,
                                          Value *Arg) {
  Value *OfType = getNormalizedPoisonValue(Ty);
  CallInst *AssignCI = nullptr;
  if (Arg->getType()->isAggregateType() && Ty->isAggregateType() &&
      allowEmitFakeUse(Arg)) {
    LLVMContext &Ctx = Arg->getContext();
    SmallVector<Metadata *, 2> ArgMDs{
        MDNode::get(Ctx, ValueAsMetadata::getConstant(OfType)),
        MDString::get(Ctx, Arg->getName())};
    B.CreateIntrinsic(Intrinsic::spv_value_md,
                      {MetadataAsValue::get(Ctx, MDTuple::get(Ctx, ArgMDs))});
    AssignCI = B.CreateIntrinsic(Intrinsic::fake_use, {Arg});
  } else {
    AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type, {Arg->getType()},
                               OfType, Arg, {}, B);
  }
  addAssignPtrTypeInstr(Arg, AssignCI);
}

void SPIRVGlobalRegistry::buildAssignPtr(IRBuilder<> &B, Type *ElemTy,
                                         Value *Arg) {
  Value *OfType = PoisonValue::get(ElemTy);
  CallInst *AssignPtrTyCI = findAssignPtrTypeInstr(Arg);
  Function *CurrF =
      B.GetInsertBlock() ? B.GetInsertBlock()->getParent() : nullptr;
  if (AssignPtrTyCI == nullptr ||
      AssignPtrTyCI->getParent()->getParent() != CurrF) {
    AssignPtrTyCI = buildIntrWithMD(
        Intrinsic::spv_assign_ptr_type, {Arg->getType()}, OfType, Arg,
        {B.getInt32(getPointerAddressSpace(Arg->getType()))}, B);
    addDeducedElementType(AssignPtrTyCI, ElemTy);
    addDeducedElementType(Arg, ElemTy);
    addAssignPtrTypeInstr(Arg, AssignPtrTyCI);
  } else {
    updateAssignType(AssignPtrTyCI, Arg, OfType);
  }
}

void SPIRVGlobalRegistry::updateAssignType(CallInst *AssignCI, Value *Arg,
                                           Value *OfType) {
  AssignCI->setArgOperand(1, buildMD(OfType));
  if (cast<IntrinsicInst>(AssignCI)->getIntrinsicID() !=
      Intrinsic::spv_assign_ptr_type)
    return;

  // update association with the pointee type
  Type *ElemTy = OfType->getType();
  addDeducedElementType(AssignCI, ElemTy);
  addDeducedElementType(Arg, ElemTy);
}

void SPIRVGlobalRegistry::addStructOffsetDecorations(
    Register Reg, StructType *Ty, MachineIRBuilder &MIRBuilder) {
  DataLayout DL;
  ArrayRef<TypeSize> Offsets = DL.getStructLayout(Ty)->getMemberOffsets();
  for (uint32_t I = 0; I < Ty->getNumElements(); ++I) {
    buildOpMemberDecorate(Reg, MIRBuilder, SPIRV::Decoration::Offset, I,
                          {static_cast<uint32_t>(Offsets[I])});
  }
}

void SPIRVGlobalRegistry::addArrayStrideDecorations(
    Register Reg, Type *ElementType, MachineIRBuilder &MIRBuilder) {
  uint32_t SizeInBytes = DataLayout().getTypeSizeInBits(ElementType) / 8;
  buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::ArrayStride,
                  {SizeInBytes});
}

bool SPIRVGlobalRegistry::hasBlockDecoration(SPIRVType *Type) const {
  Register Def = getSPIRVTypeID(Type);
  for (const MachineInstr &Use :
       Type->getMF()->getRegInfo().use_instructions(Def)) {
    if (Use.getOpcode() != SPIRV::OpDecorate)
      continue;

    if (Use.getOperand(1).getImm() == SPIRV::Decoration::Block)
      return true;
  }
  return false;
}
