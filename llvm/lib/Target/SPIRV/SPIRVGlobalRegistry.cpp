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

using namespace llvm;
SPIRVGlobalRegistry::SPIRVGlobalRegistry(unsigned PointerSize)
    : PointerSize(PointerSize) {}

SPIRVType *SPIRVGlobalRegistry::assignIntTypeToVReg(unsigned BitWidth,
                                                    Register VReg,
                                                    MachineInstr &I,
                                                    const SPIRVInstrInfo &TII) {
  SPIRVType *SpirvType = getOrCreateSPIRVIntegerType(BitWidth, I, TII);
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
                                                MachineFunction &MF) {
  VRegToTypeMap[&MF][VReg] = SpirvType;
}

static Register createTypeVReg(MachineIRBuilder &MIRBuilder) {
  auto &MRI = MIRBuilder.getMF().getRegInfo();
  auto Res = MRI.createGenericVirtualRegister(LLT::scalar(32));
  MRI.setRegClass(Res, &SPIRV::TYPERegClass);
  return Res;
}

static Register createTypeVReg(MachineRegisterInfo &MRI) {
  auto Res = MRI.createGenericVirtualRegister(LLT::scalar(32));
  MRI.setRegClass(Res, &SPIRV::TYPERegClass);
  return Res;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeBool(MachineIRBuilder &MIRBuilder) {
  return MIRBuilder.buildInstr(SPIRV::OpTypeBool)
      .addDef(createTypeVReg(MIRBuilder));
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeInt(uint32_t Width,
                                             MachineIRBuilder &MIRBuilder,
                                             bool IsSigned) {
  assert(Width <= 64 && "Unsupported integer width!");
  const SPIRVSubtarget &ST =
      cast<SPIRVSubtarget>(MIRBuilder.getMF().getSubtarget());
  if (ST.canUseExtension(
          SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers)) {
    MIRBuilder.buildInstr(SPIRV::OpExtension)
        .addImm(SPIRV::Extension::SPV_INTEL_arbitrary_precision_integers);
    MIRBuilder.buildInstr(SPIRV::OpCapability)
        .addImm(SPIRV::Capability::ArbitraryPrecisionIntegersINTEL);
  } else if (Width <= 8)
    Width = 8;
  else if (Width <= 16)
    Width = 16;
  else if (Width <= 32)
    Width = 32;
  else if (Width <= 64)
    Width = 64;

  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeInt)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addImm(Width)
                 .addImm(IsSigned ? 1 : 0);
  return MIB;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeFloat(uint32_t Width,
                                               MachineIRBuilder &MIRBuilder) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeFloat)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addImm(Width);
  return MIB;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeVoid(MachineIRBuilder &MIRBuilder) {
  return MIRBuilder.buildInstr(SPIRV::OpTypeVoid)
      .addDef(createTypeVReg(MIRBuilder));
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeVector(uint32_t NumElems,
                                                SPIRVType *ElemType,
                                                MachineIRBuilder &MIRBuilder) {
  auto EleOpc = ElemType->getOpcode();
  assert((EleOpc == SPIRV::OpTypeInt || EleOpc == SPIRV::OpTypeFloat ||
          EleOpc == SPIRV::OpTypeBool) &&
         "Invalid vector element type");

  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeVector)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addUse(getSPIRVTypeID(ElemType))
                 .addImm(NumElems);
  return MIB;
}

std::tuple<Register, ConstantInt *, bool>
SPIRVGlobalRegistry::getOrCreateConstIntReg(uint64_t Val, SPIRVType *SpvType,
                                            MachineIRBuilder *MIRBuilder,
                                            MachineInstr *I,
                                            const SPIRVInstrInfo *TII) {
  const IntegerType *LLVMIntTy;
  if (SpvType)
    LLVMIntTy = cast<IntegerType>(getTypeForSPIRVType(SpvType));
  else
    LLVMIntTy = IntegerType::getInt32Ty(CurMF->getFunction().getContext());
  bool NewInstr = false;
  // Find a constant in DT or build a new one.
  ConstantInt *CI = ConstantInt::get(const_cast<IntegerType *>(LLVMIntTy), Val);
  Register Res = DT.find(CI, CurMF);
  if (!Res.isValid()) {
    unsigned BitWidth = SpvType ? getScalarOrVectorBitWidth(SpvType) : 32;
    LLT LLTy = LLT::scalar(32);
    Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
    CurMF->getRegInfo().setRegClass(Res, &SPIRV::IDRegClass);
    if (MIRBuilder)
      assignTypeToVReg(LLVMIntTy, Res, *MIRBuilder);
    else
      assignIntTypeToVReg(BitWidth, Res, *I, *TII);
    DT.add(CI, CurMF, Res);
    NewInstr = true;
  }
  return std::make_tuple(Res, CI, NewInstr);
}

Register SPIRVGlobalRegistry::getOrCreateConstInt(uint64_t Val, MachineInstr &I,
                                                  SPIRVType *SpvType,
                                                  const SPIRVInstrInfo &TII) {
  assert(SpvType);
  ConstantInt *CI;
  Register Res;
  bool New;
  std::tie(Res, CI, New) =
      getOrCreateConstIntReg(Val, SpvType, nullptr, &I, &TII);
  // If we have found Res register which is defined by the passed G_CONSTANT
  // machine instruction, a new constant instruction should be created.
  if (!New && (!I.getOperand(0).isReg() || Res != I.getOperand(0).getReg()))
    return Res;
  MachineInstrBuilder MIB;
  MachineBasicBlock &BB = *I.getParent();
  if (Val) {
    MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpConstantI))
              .addDef(Res)
              .addUse(getSPIRVTypeID(SpvType));
    addNumImm(APInt(getScalarOrVectorBitWidth(SpvType), Val), MIB);
  } else {
    MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpConstantNull))
              .addDef(Res)
              .addUse(getSPIRVTypeID(SpvType));
  }
  const auto &ST = CurMF->getSubtarget();
  constrainSelectedInstRegOperands(*MIB, *ST.getInstrInfo(),
                                   *ST.getRegisterInfo(), *ST.getRegBankInfo());
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantInt(uint64_t Val,
                                               MachineIRBuilder &MIRBuilder,
                                               SPIRVType *SpvType,
                                               bool EmitIR) {
  auto &MF = MIRBuilder.getMF();
  const IntegerType *LLVMIntTy;
  if (SpvType)
    LLVMIntTy = cast<IntegerType>(getTypeForSPIRVType(SpvType));
  else
    LLVMIntTy = IntegerType::getInt32Ty(MF.getFunction().getContext());
  // Find a constant in DT or build a new one.
  const auto ConstInt =
      ConstantInt::get(const_cast<IntegerType *>(LLVMIntTy), Val);
  Register Res = DT.find(ConstInt, &MF);
  if (!Res.isValid()) {
    unsigned BitWidth = SpvType ? getScalarOrVectorBitWidth(SpvType) : 32;
    LLT LLTy = LLT::scalar(EmitIR ? BitWidth : 32);
    Res = MF.getRegInfo().createGenericVirtualRegister(LLTy);
    MF.getRegInfo().setRegClass(Res, &SPIRV::IDRegClass);
    assignTypeToVReg(LLVMIntTy, Res, MIRBuilder,
                     SPIRV::AccessQualifier::ReadWrite, EmitIR);
    DT.add(ConstInt, &MIRBuilder.getMF(), Res);
    if (EmitIR) {
      MIRBuilder.buildConstant(Res, *ConstInt);
    } else {
      MachineInstrBuilder MIB;
      if (Val) {
        assert(SpvType);
        MIB = MIRBuilder.buildInstr(SPIRV::OpConstantI)
                  .addDef(Res)
                  .addUse(getSPIRVTypeID(SpvType));
        addNumImm(APInt(BitWidth, Val), MIB);
      } else {
        assert(SpvType);
        MIB = MIRBuilder.buildInstr(SPIRV::OpConstantNull)
                  .addDef(Res)
                  .addUse(getSPIRVTypeID(SpvType));
      }
      const auto &Subtarget = CurMF->getSubtarget();
      constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                       *Subtarget.getRegisterInfo(),
                                       *Subtarget.getRegBankInfo());
    }
  }
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantFP(APFloat Val,
                                              MachineIRBuilder &MIRBuilder,
                                              SPIRVType *SpvType) {
  auto &MF = MIRBuilder.getMF();
  auto &Ctx = MF.getFunction().getContext();
  if (!SpvType) {
    const Type *LLVMFPTy = Type::getFloatTy(Ctx);
    SpvType = getOrCreateSPIRVType(LLVMFPTy, MIRBuilder);
  }
  // Find a constant in DT or build a new one.
  const auto ConstFP = ConstantFP::get(Ctx, Val);
  Register Res = DT.find(ConstFP, &MF);
  if (!Res.isValid()) {
    Res = MF.getRegInfo().createGenericVirtualRegister(LLT::scalar(32));
    MF.getRegInfo().setRegClass(Res, &SPIRV::IDRegClass);
    assignSPIRVTypeToVReg(SpvType, Res, MF);
    DT.add(ConstFP, &MF, Res);

    MachineInstrBuilder MIB;
    MIB = MIRBuilder.buildInstr(SPIRV::OpConstantF)
              .addDef(Res)
              .addUse(getSPIRVTypeID(SpvType));
    addNumImm(ConstFP->getValueAPF().bitcastToAPInt(), MIB);
  }

  return Res;
}

Register SPIRVGlobalRegistry::getOrCreateIntCompositeOrNull(
    uint64_t Val, MachineInstr &I, SPIRVType *SpvType,
    const SPIRVInstrInfo &TII, Constant *CA, unsigned BitWidth,
    unsigned ElemCnt) {
  // Find a constant vector in DT or build a new one.
  Register Res = DT.find(CA, CurMF);
  if (!Res.isValid()) {
    SPIRVType *SpvBaseType = getOrCreateSPIRVIntegerType(BitWidth, I, TII);
    // SpvScalConst should be created before SpvVecConst to avoid undefined ID
    // error on validation.
    // TODO: can moved below once sorting of types/consts/defs is implemented.
    Register SpvScalConst;
    if (Val)
      SpvScalConst = getOrCreateConstInt(Val, I, SpvBaseType, TII);
    // TODO: maybe use bitwidth of base type.
    LLT LLTy = LLT::scalar(32);
    Register SpvVecConst =
        CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
    CurMF->getRegInfo().setRegClass(SpvVecConst, &SPIRV::IDRegClass);
    assignSPIRVTypeToVReg(SpvType, SpvVecConst, *CurMF);
    DT.add(CA, CurMF, SpvVecConst);
    MachineInstrBuilder MIB;
    MachineBasicBlock &BB = *I.getParent();
    if (Val) {
      MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpConstantComposite))
                .addDef(SpvVecConst)
                .addUse(getSPIRVTypeID(SpvType));
      for (unsigned i = 0; i < ElemCnt; ++i)
        MIB.addUse(SpvScalConst);
    } else {
      MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpConstantNull))
                .addDef(SpvVecConst)
                .addUse(getSPIRVTypeID(SpvType));
    }
    const auto &Subtarget = CurMF->getSubtarget();
    constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                     *Subtarget.getRegisterInfo(),
                                     *Subtarget.getRegBankInfo());
    return SpvVecConst;
  }
  return Res;
}

Register
SPIRVGlobalRegistry::getOrCreateConsIntVector(uint64_t Val, MachineInstr &I,
                                              SPIRVType *SpvType,
                                              const SPIRVInstrInfo &TII) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isVectorTy());
  const FixedVectorType *LLVMVecTy = cast<FixedVectorType>(LLVMTy);
  Type *LLVMBaseTy = LLVMVecTy->getElementType();
  const auto ConstInt = ConstantInt::get(LLVMBaseTy, Val);
  auto ConstVec =
      ConstantVector::getSplat(LLVMVecTy->getElementCount(), ConstInt);
  unsigned BW = getScalarOrVectorBitWidth(SpvType);
  return getOrCreateIntCompositeOrNull(Val, I, SpvType, TII, ConstVec, BW,
                                       SpvType->getOperand(2).getImm());
}

Register
SPIRVGlobalRegistry::getOrCreateConsIntArray(uint64_t Val, MachineInstr &I,
                                             SPIRVType *SpvType,
                                             const SPIRVInstrInfo &TII) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isArrayTy());
  const ArrayType *LLVMArrTy = cast<ArrayType>(LLVMTy);
  Type *LLVMBaseTy = LLVMArrTy->getElementType();
  const auto ConstInt = ConstantInt::get(LLVMBaseTy, Val);
  auto ConstArr =
      ConstantArray::get(const_cast<ArrayType *>(LLVMArrTy), {ConstInt});
  SPIRVType *SpvBaseTy = getSPIRVTypeForVReg(SpvType->getOperand(1).getReg());
  unsigned BW = getScalarOrVectorBitWidth(SpvBaseTy);
  return getOrCreateIntCompositeOrNull(Val, I, SpvType, TII, ConstArr, BW,
                                       LLVMArrTy->getNumElements());
}

Register SPIRVGlobalRegistry::getOrCreateIntCompositeOrNull(
    uint64_t Val, MachineIRBuilder &MIRBuilder, SPIRVType *SpvType, bool EmitIR,
    Constant *CA, unsigned BitWidth, unsigned ElemCnt) {
  Register Res = DT.find(CA, CurMF);
  if (!Res.isValid()) {
    Register SpvScalConst;
    if (Val || EmitIR) {
      SPIRVType *SpvBaseType =
          getOrCreateSPIRVIntegerType(BitWidth, MIRBuilder);
      SpvScalConst = buildConstantInt(Val, MIRBuilder, SpvBaseType, EmitIR);
    }
    LLT LLTy = EmitIR ? LLT::fixed_vector(ElemCnt, BitWidth) : LLT::scalar(32);
    Register SpvVecConst =
        CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
    CurMF->getRegInfo().setRegClass(SpvVecConst, &SPIRV::IDRegClass);
    assignSPIRVTypeToVReg(SpvType, SpvVecConst, *CurMF);
    DT.add(CA, CurMF, SpvVecConst);
    if (EmitIR) {
      MIRBuilder.buildSplatVector(SpvVecConst, SpvScalConst);
    } else {
      if (Val) {
        auto MIB = MIRBuilder.buildInstr(SPIRV::OpConstantComposite)
                       .addDef(SpvVecConst)
                       .addUse(getSPIRVTypeID(SpvType));
        for (unsigned i = 0; i < ElemCnt; ++i)
          MIB.addUse(SpvScalConst);
      } else {
        MIRBuilder.buildInstr(SPIRV::OpConstantNull)
            .addDef(SpvVecConst)
            .addUse(getSPIRVTypeID(SpvType));
      }
    }
    return SpvVecConst;
  }
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
SPIRVGlobalRegistry::getOrCreateConsIntArray(uint64_t Val,
                                             MachineIRBuilder &MIRBuilder,
                                             SPIRVType *SpvType, bool EmitIR) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy->isArrayTy());
  const ArrayType *LLVMArrTy = cast<ArrayType>(LLVMTy);
  Type *LLVMBaseTy = LLVMArrTy->getElementType();
  const auto ConstInt = ConstantInt::get(LLVMBaseTy, Val);
  auto ConstArr =
      ConstantArray::get(const_cast<ArrayType *>(LLVMArrTy), {ConstInt});
  SPIRVType *SpvBaseTy = getSPIRVTypeForVReg(SpvType->getOperand(1).getReg());
  unsigned BW = getScalarOrVectorBitWidth(SpvBaseTy);
  return getOrCreateIntCompositeOrNull(Val, MIRBuilder, SpvType, EmitIR,
                                       ConstArr, BW,
                                       LLVMArrTy->getNumElements());
}

Register
SPIRVGlobalRegistry::getOrCreateConstNullPtr(MachineIRBuilder &MIRBuilder,
                                             SPIRVType *SpvType) {
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  const PointerType *LLVMPtrTy = cast<PointerType>(LLVMTy);
  // Find a constant in DT or build a new one.
  Constant *CP = ConstantPointerNull::get(const_cast<PointerType *>(LLVMPtrTy));
  Register Res = DT.find(CP, CurMF);
  if (!Res.isValid()) {
    LLT LLTy = LLT::pointer(LLVMPtrTy->getAddressSpace(), PointerSize);
    Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
    CurMF->getRegInfo().setRegClass(Res, &SPIRV::IDRegClass);
    assignSPIRVTypeToVReg(SpvType, Res, *CurMF);
    MIRBuilder.buildInstr(SPIRV::OpConstantNull)
        .addDef(Res)
        .addUse(getSPIRVTypeID(SpvType));
    DT.add(CP, CurMF, Res);
  }
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantSampler(
    Register ResReg, unsigned AddrMode, unsigned Param, unsigned FilerMode,
    MachineIRBuilder &MIRBuilder, SPIRVType *SpvType) {
  SPIRVType *SampTy;
  if (SpvType)
    SampTy = getOrCreateSPIRVType(getTypeForSPIRVType(SpvType), MIRBuilder);
  else
    SampTy = getOrCreateSPIRVTypeByName("opencl.sampler_t", MIRBuilder);

  auto Sampler =
      ResReg.isValid()
          ? ResReg
          : MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  auto Res = MIRBuilder.buildInstr(SPIRV::OpConstantSampler)
                 .addDef(Sampler)
                 .addUse(getSPIRVTypeID(SampTy))
                 .addImm(AddrMode)
                 .addImm(Param)
                 .addImm(FilerMode);
  assert(Res->getOperand(0).isReg());
  return Res->getOperand(0).getReg();
}

Register SPIRVGlobalRegistry::buildGlobalVariable(
    Register ResVReg, SPIRVType *BaseType, StringRef Name,
    const GlobalValue *GV, SPIRV::StorageClass::StorageClass Storage,
    const MachineInstr *Init, bool IsConst, bool HasLinkageTy,
    SPIRV::LinkageType::LinkageType LinkageType, MachineIRBuilder &MIRBuilder,
    bool IsInstSelector) {
  const GlobalVariable *GVar = nullptr;
  if (GV)
    GVar = cast<const GlobalVariable>(GV);
  else {
    // If GV is not passed explicitly, use the name to find or construct
    // the global variable.
    Module *M = MIRBuilder.getMF().getFunction().getParent();
    GVar = M->getGlobalVariable(Name);
    if (GVar == nullptr) {
      const Type *Ty = getTypeForSPIRVType(BaseType); // TODO: check type.
      GVar = new GlobalVariable(*M, const_cast<Type *>(Ty), false,
                                GlobalValue::ExternalLinkage, nullptr,
                                Twine(Name));
    }
    GV = GVar;
  }
  Register Reg = DT.find(GVar, &MIRBuilder.getMF());
  if (Reg.isValid()) {
    if (Reg != ResVReg)
      MIRBuilder.buildCopy(ResVReg, Reg);
    return ResVReg;
  }

  auto MIB = MIRBuilder.buildInstr(SPIRV::OpVariable)
                 .addDef(ResVReg)
                 .addUse(getSPIRVTypeID(BaseType))
                 .addImm(static_cast<uint32_t>(Storage));

  if (Init != 0) {
    MIB.addUse(Init->getOperand(0).getReg());
  }

  // ISel may introduce a new register on this step, so we need to add it to
  // DT and correct its type avoiding fails on the next stage.
  if (IsInstSelector) {
    const auto &Subtarget = CurMF->getSubtarget();
    constrainSelectedInstRegOperands(*MIB, *Subtarget.getInstrInfo(),
                                     *Subtarget.getRegisterInfo(),
                                     *Subtarget.getRegBankInfo());
  }
  Reg = MIB->getOperand(0).getReg();
  DT.add(GVar, &MIRBuilder.getMF(), Reg);

  // Set to Reg the same type as ResVReg has.
  auto MRI = MIRBuilder.getMRI();
  assert(MRI->getType(ResVReg).isPointer() && "Pointer type is expected");
  if (Reg != ResVReg) {
    LLT RegLLTy = LLT::pointer(MRI->getType(ResVReg).getAddressSpace(), 32);
    MRI->setType(Reg, RegLLTy);
    assignSPIRVTypeToVReg(BaseType, Reg, MIRBuilder.getMF());
  }

  // If it's a global variable with name, output OpName for it.
  if (GVar && GVar->hasName())
    buildOpName(Reg, GVar->getName(), MIRBuilder);

  // Output decorations for the GV.
  // TODO: maybe move to GenerateDecorations pass.
  if (IsConst)
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::Constant, {});

  if (GVar && GVar->getAlign().valueOrOne().value() != 1) {
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

  return Reg;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeArray(uint32_t NumElems,
                                               SPIRVType *ElemType,
                                               MachineIRBuilder &MIRBuilder,
                                               bool EmitIR) {
  assert((ElemType->getOpcode() != SPIRV::OpTypeVoid) &&
         "Invalid array element type");
  Register NumElementsVReg =
      buildConstantInt(NumElems, MIRBuilder, nullptr, EmitIR);
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeArray)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addUse(getSPIRVTypeID(ElemType))
                 .addUse(NumElementsVReg);
  return MIB;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeOpaque(const StructType *Ty,
                                                MachineIRBuilder &MIRBuilder) {
  assert(Ty->hasName());
  const StringRef Name = Ty->hasName() ? Ty->getName() : "";
  Register ResVReg = createTypeVReg(MIRBuilder);
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeOpaque).addDef(ResVReg);
  addStringImm(Name, MIB);
  buildOpName(ResVReg, Name, MIRBuilder);
  return MIB;
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeStruct(const StructType *Ty,
                                                MachineIRBuilder &MIRBuilder,
                                                bool EmitIR) {
  SmallVector<Register, 4> FieldTypes;
  for (const auto &Elem : Ty->elements()) {
    SPIRVType *ElemTy = findSPIRVType(Elem, MIRBuilder);
    assert(ElemTy && ElemTy->getOpcode() != SPIRV::OpTypeVoid &&
           "Invalid struct element type");
    FieldTypes.push_back(getSPIRVTypeID(ElemTy));
  }
  Register ResVReg = createTypeVReg(MIRBuilder);
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeStruct).addDef(ResVReg);
  for (const auto &Ty : FieldTypes)
    MIB.addUse(Ty);
  if (Ty->hasName())
    buildOpName(ResVReg, Ty->getName(), MIRBuilder);
  if (Ty->isPacked())
    buildOpDecorate(ResVReg, MIRBuilder, SPIRV::Decoration::CPacked, {});
  return MIB;
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
  return MIRBuilder.buildInstr(SPIRV::OpTypePointer)
      .addDef(Reg)
      .addImm(static_cast<uint32_t>(SC))
      .addUse(getSPIRVTypeID(ElemType));
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeForwardPointer(
    SPIRV::StorageClass::StorageClass SC, MachineIRBuilder &MIRBuilder) {
  return MIRBuilder.buildInstr(SPIRV::OpTypeForwardPointer)
      .addUse(createTypeVReg(MIRBuilder))
      .addImm(static_cast<uint32_t>(SC));
}

SPIRVType *SPIRVGlobalRegistry::getOpTypeFunction(
    SPIRVType *RetType, const SmallVectorImpl<SPIRVType *> &ArgTypes,
    MachineIRBuilder &MIRBuilder) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypeFunction)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addUse(getSPIRVTypeID(RetType));
  for (const SPIRVType *ArgType : ArgTypes)
    MIB.addUse(getSPIRVTypeID(ArgType));
  return MIB;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeFunctionWithArgs(
    const Type *Ty, SPIRVType *RetType,
    const SmallVectorImpl<SPIRVType *> &ArgTypes,
    MachineIRBuilder &MIRBuilder) {
  Register Reg = DT.find(Ty, &MIRBuilder.getMF());
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  SPIRVType *SpirvType = getOpTypeFunction(RetType, ArgTypes, MIRBuilder);
  DT.add(Ty, CurMF, getSPIRVTypeID(SpirvType));
  return finishCreatingSPIRVType(Ty, SpirvType);
}

SPIRVType *SPIRVGlobalRegistry::findSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual, bool EmitIR) {
  Register Reg = DT.find(Ty, &MIRBuilder.getMF());
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  if (ForwardPointerTypes.contains(Ty))
    return ForwardPointerTypes[Ty];
  return restOfCreateSPIRVType(Ty, MIRBuilder, AccQual, EmitIR);
}

Register SPIRVGlobalRegistry::getSPIRVTypeID(const SPIRVType *SpirvType) const {
  assert(SpirvType && "Attempting to get type id for nullptr type.");
  if (SpirvType->getOpcode() == SPIRV::OpTypeForwardPointer)
    return SpirvType->uses().begin()->getReg();
  return SpirvType->defs().begin()->getReg();
}

SPIRVType *SPIRVGlobalRegistry::createSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccQual, bool EmitIR) {
  if (isSpecialOpaqueType(Ty))
    return getOrCreateSpecialType(Ty, MIRBuilder, AccQual);
  auto &TypeToSPIRVTypeMap = DT.getTypes()->getAllUses();
  auto t = TypeToSPIRVTypeMap.find(Ty);
  if (t != TypeToSPIRVTypeMap.end()) {
    auto tt = t->second.find(&MIRBuilder.getMF());
    if (tt != t->second.end())
      return getSPIRVTypeForVReg(tt->second);
  }

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
        findSPIRVType(cast<FixedVectorType>(Ty)->getElementType(), MIRBuilder);
    return getOpTypeVector(cast<FixedVectorType>(Ty)->getNumElements(), El,
                           MIRBuilder);
  }
  if (Ty->isArrayTy()) {
    SPIRVType *El = findSPIRVType(Ty->getArrayElementType(), MIRBuilder);
    return getOpTypeArray(Ty->getArrayNumElements(), El, MIRBuilder, EmitIR);
  }
  if (auto SType = dyn_cast<StructType>(Ty)) {
    if (SType->isOpaque())
      return getOpTypeOpaque(SType, MIRBuilder);
    return getOpTypeStruct(SType, MIRBuilder, EmitIR);
  }
  if (auto FType = dyn_cast<FunctionType>(Ty)) {
    SPIRVType *RetTy = findSPIRVType(FType->getReturnType(), MIRBuilder);
    SmallVector<SPIRVType *, 4> ParamTypes;
    for (const auto &t : FType->params()) {
      ParamTypes.push_back(findSPIRVType(t, MIRBuilder));
    }
    return getOpTypeFunction(RetTy, ParamTypes, MIRBuilder);
  }
  if (auto PType = dyn_cast<PointerType>(Ty)) {
    SPIRVType *SpvElementType;
    // At the moment, all opaque pointers correspond to i8 element type.
    // TODO: change the implementation once opaque pointers are supported
    // in the SPIR-V specification.
    SpvElementType = getOrCreateSPIRVIntegerType(8, MIRBuilder);
    auto SC = addressSpaceToStorageClass(PType->getAddressSpace());
    // Null pointer means we have a loop in type definitions, make and
    // return corresponding OpTypeForwardPointer.
    if (SpvElementType == nullptr) {
      if (!ForwardPointerTypes.contains(Ty))
        ForwardPointerTypes[PType] = getOpTypeForwardPointer(SC, MIRBuilder);
      return ForwardPointerTypes[PType];
    }
    // If we have forward pointer associated with this type, use its register
    // operand to create OpTypePointer.
    if (ForwardPointerTypes.contains(PType)) {
      Register Reg = getSPIRVTypeID(ForwardPointerTypes[PType]);
      return getOpTypePointer(SC, SpvElementType, MIRBuilder, Reg);
    }

    return getOrCreateSPIRVPointerType(SpvElementType, MIRBuilder, SC);
  }
  llvm_unreachable("Unable to convert LLVM type to SPIRVType");
}

SPIRVType *SPIRVGlobalRegistry::restOfCreateSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual, bool EmitIR) {
  if (TypesInProcessing.count(Ty) && !Ty->isPointerTy())
    return nullptr;
  TypesInProcessing.insert(Ty);
  SPIRVType *SpirvType = createSPIRVType(Ty, MIRBuilder, AccessQual, EmitIR);
  TypesInProcessing.erase(Ty);
  VRegToTypeMap[&MIRBuilder.getMF()][getSPIRVTypeID(SpirvType)] = SpirvType;
  SPIRVToLLVMType[SpirvType] = Ty;
  Register Reg = DT.find(Ty, &MIRBuilder.getMF());
  // Do not add OpTypeForwardPointer to DT, a corresponding normal pointer type
  // will be added later. For special types it is already added to DT.
  if (SpirvType->getOpcode() != SPIRV::OpTypeForwardPointer && !Reg.isValid() &&
      !isSpecialOpaqueType(Ty)) {
    if (!Ty->isPointerTy())
      DT.add(Ty, &MIRBuilder.getMF(), getSPIRVTypeID(SpirvType));
    else
      DT.add(Type::getInt8Ty(MIRBuilder.getMF().getFunction().getContext()),
             Ty->getPointerAddressSpace(), &MIRBuilder.getMF(),
             getSPIRVTypeID(SpirvType));
  }

  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::getSPIRVTypeForVReg(Register VReg) const {
  auto t = VRegToTypeMap.find(CurMF);
  if (t != VRegToTypeMap.end()) {
    auto tt = t->second.find(VReg);
    if (tt != t->second.end())
      return tt->second;
  }
  return nullptr;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVType(
    const Type *Ty, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual, bool EmitIR) {
  Register Reg;
  if (!Ty->isPointerTy())
    Reg = DT.find(Ty, &MIRBuilder.getMF());
  else
    Reg =
        DT.find(Type::getInt8Ty(MIRBuilder.getMF().getFunction().getContext()),
                Ty->getPointerAddressSpace(), &MIRBuilder.getMF());

  if (Reg.isValid() && !isSpecialOpaqueType(Ty))
    return getSPIRVTypeForVReg(Reg);
  TypesInProcessing.clear();
  SPIRVType *STy = restOfCreateSPIRVType(Ty, MIRBuilder, AccessQual, EmitIR);
  // Create normal pointer types for the corresponding OpTypeForwardPointers.
  for (auto &CU : ForwardPointerTypes) {
    const Type *Ty2 = CU.first;
    SPIRVType *STy2 = CU.second;
    if ((Reg = DT.find(Ty2, &MIRBuilder.getMF())).isValid())
      STy2 = getSPIRVTypeForVReg(Reg);
    else
      STy2 = restOfCreateSPIRVType(Ty2, MIRBuilder, AccessQual, EmitIR);
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

bool SPIRVGlobalRegistry::isScalarOrVectorSigned(const SPIRVType *Type) const {
  assert(Type && "Invalid Type pointer");
  if (Type->getOpcode() == SPIRV::OpTypeVector) {
    auto EleTypeReg = Type->getOperand(1).getReg();
    Type = getSPIRVTypeForVReg(EleTypeReg);
  }
  if (Type->getOpcode() == SPIRV::OpTypeInt)
    return Type->getOperand(2).getImm() != 0;
  llvm_unreachable("Attempting to get sign of non-integer type.");
}

SPIRV::StorageClass::StorageClass
SPIRVGlobalRegistry::getPointerStorageClass(Register VReg) const {
  SPIRVType *Type = getSPIRVTypeForVReg(VReg);
  assert(Type && Type->getOpcode() == SPIRV::OpTypePointer &&
         Type->getOperand(1).isImm() && "Pointer type is expected");
  return static_cast<SPIRV::StorageClass::StorageClass>(
      Type->getOperand(1).getImm());
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeImage(
    MachineIRBuilder &MIRBuilder, SPIRVType *SampledType, SPIRV::Dim::Dim Dim,
    uint32_t Depth, uint32_t Arrayed, uint32_t Multisampled, uint32_t Sampled,
    SPIRV::ImageFormat::ImageFormat ImageFormat,
    SPIRV::AccessQualifier::AccessQualifier AccessQual) {
  SPIRV::ImageTypeDescriptor TD(SPIRVToLLVMType.lookup(SampledType), Dim, Depth,
                                Arrayed, Multisampled, Sampled, ImageFormat,
                                AccessQual);
  if (auto *Res = checkSpecialInstr(TD, MIRBuilder))
    return Res;
  Register ResVReg = createTypeVReg(MIRBuilder);
  DT.add(TD, &MIRBuilder.getMF(), ResVReg);
  return MIRBuilder.buildInstr(SPIRV::OpTypeImage)
      .addDef(ResVReg)
      .addUse(getSPIRVTypeID(SampledType))
      .addImm(Dim)
      .addImm(Depth)        // Depth (whether or not it is a Depth image).
      .addImm(Arrayed)      // Arrayed.
      .addImm(Multisampled) // Multisampled (0 = only single-sample).
      .addImm(Sampled)      // Sampled (0 = usage known at runtime).
      .addImm(ImageFormat)
      .addImm(AccessQual);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateOpTypeSampler(MachineIRBuilder &MIRBuilder) {
  SPIRV::SamplerTypeDescriptor TD;
  if (auto *Res = checkSpecialInstr(TD, MIRBuilder))
    return Res;
  Register ResVReg = createTypeVReg(MIRBuilder);
  DT.add(TD, &MIRBuilder.getMF(), ResVReg);
  return MIRBuilder.buildInstr(SPIRV::OpTypeSampler).addDef(ResVReg);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypePipe(
    MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual) {
  SPIRV::PipeTypeDescriptor TD(AccessQual);
  if (auto *Res = checkSpecialInstr(TD, MIRBuilder))
    return Res;
  Register ResVReg = createTypeVReg(MIRBuilder);
  DT.add(TD, &MIRBuilder.getMF(), ResVReg);
  return MIRBuilder.buildInstr(SPIRV::OpTypePipe)
      .addDef(ResVReg)
      .addImm(AccessQual);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeDeviceEvent(
    MachineIRBuilder &MIRBuilder) {
  SPIRV::DeviceEventTypeDescriptor TD;
  if (auto *Res = checkSpecialInstr(TD, MIRBuilder))
    return Res;
  Register ResVReg = createTypeVReg(MIRBuilder);
  DT.add(TD, &MIRBuilder.getMF(), ResVReg);
  return MIRBuilder.buildInstr(SPIRV::OpTypeDeviceEvent).addDef(ResVReg);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeSampledImage(
    SPIRVType *ImageType, MachineIRBuilder &MIRBuilder) {
  SPIRV::SampledImageTypeDescriptor TD(
      SPIRVToLLVMType.lookup(MIRBuilder.getMF().getRegInfo().getVRegDef(
          ImageType->getOperand(1).getReg())),
      ImageType);
  if (auto *Res = checkSpecialInstr(TD, MIRBuilder))
    return Res;
  Register ResVReg = createTypeVReg(MIRBuilder);
  DT.add(TD, &MIRBuilder.getMF(), ResVReg);
  return MIRBuilder.buildInstr(SPIRV::OpTypeSampledImage)
      .addDef(ResVReg)
      .addUse(getSPIRVTypeID(ImageType));
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateOpTypeByOpcode(
    const Type *Ty, MachineIRBuilder &MIRBuilder, unsigned Opcode) {
  Register ResVReg = DT.find(Ty, &MIRBuilder.getMF());
  if (ResVReg.isValid())
    return MIRBuilder.getMF().getRegInfo().getUniqueVRegDef(ResVReg);
  ResVReg = createTypeVReg(MIRBuilder);
  SPIRVType *SpirvTy = MIRBuilder.buildInstr(Opcode).addDef(ResVReg);
  DT.add(Ty, &MIRBuilder.getMF(), ResVReg);
  return SpirvTy;
}

const MachineInstr *
SPIRVGlobalRegistry::checkSpecialInstr(const SPIRV::SpecialTypeDescriptor &TD,
                                       MachineIRBuilder &MIRBuilder) {
  Register Reg = DT.find(TD, &MIRBuilder.getMF());
  if (Reg.isValid())
    return MIRBuilder.getMF().getRegInfo().getUniqueVRegDef(Reg);
  return nullptr;
}

// TODO: maybe use tablegen to implement this.
SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVTypeByName(
    StringRef TypeStr, MachineIRBuilder &MIRBuilder,
    SPIRV::StorageClass::StorageClass SC,
    SPIRV::AccessQualifier::AccessQualifier AQ) {
  unsigned VecElts = 0;
  auto &Ctx = MIRBuilder.getMF().getFunction().getContext();

  // Parse strings representing either a SPIR-V or OpenCL builtin type.
  if (hasBuiltinTypePrefix(TypeStr))
    return getOrCreateSPIRVType(
        SPIRV::parseBuiltinTypeNameToTargetExtType(TypeStr.str(), MIRBuilder),
        MIRBuilder, AQ);

  // Parse type name in either "typeN" or "type vector[N]" format, where
  // N is the number of elements of the vector.
  Type *Ty;

  TypeStr.consume_front("atomic_");

  if (TypeStr.starts_with("void")) {
    Ty = Type::getVoidTy(Ctx);
    TypeStr = TypeStr.substr(strlen("void"));
  } else if (TypeStr.starts_with("bool")) {
    Ty = Type::getIntNTy(Ctx, 1);
    TypeStr = TypeStr.substr(strlen("bool"));
  } else if (TypeStr.starts_with("char") || TypeStr.starts_with("uchar")) {
    Ty = Type::getInt8Ty(Ctx);
    TypeStr = TypeStr.starts_with("char") ? TypeStr.substr(strlen("char"))
                                          : TypeStr.substr(strlen("uchar"));
  } else if (TypeStr.starts_with("short") || TypeStr.starts_with("ushort")) {
    Ty = Type::getInt16Ty(Ctx);
    TypeStr = TypeStr.starts_with("short") ? TypeStr.substr(strlen("short"))
                                           : TypeStr.substr(strlen("ushort"));
  } else if (TypeStr.starts_with("int") || TypeStr.starts_with("uint")) {
    Ty = Type::getInt32Ty(Ctx);
    TypeStr = TypeStr.starts_with("int") ? TypeStr.substr(strlen("int"))
                                         : TypeStr.substr(strlen("uint"));
  } else if (TypeStr.starts_with("long") || TypeStr.starts_with("ulong")) {
    Ty = Type::getInt64Ty(Ctx);
    TypeStr = TypeStr.starts_with("long") ? TypeStr.substr(strlen("long"))
                                          : TypeStr.substr(strlen("ulong"));
  } else if (TypeStr.starts_with("half")) {
    Ty = Type::getHalfTy(Ctx);
    TypeStr = TypeStr.substr(strlen("half"));
  } else if (TypeStr.starts_with("float")) {
    Ty = Type::getFloatTy(Ctx);
    TypeStr = TypeStr.substr(strlen("float"));
  } else if (TypeStr.starts_with("double")) {
    Ty = Type::getDoubleTy(Ctx);
    TypeStr = TypeStr.substr(strlen("double"));
  } else
    llvm_unreachable("Unable to recognize SPIRV type name.");

  auto SpirvTy = getOrCreateSPIRVType(Ty, MIRBuilder, AQ);

  // Handle "type*" or  "type* vector[N]".
  if (TypeStr.starts_with("*")) {
    SpirvTy = getOrCreateSPIRVPointerType(SpirvTy, MIRBuilder, SC);
    TypeStr = TypeStr.substr(strlen("*"));
  }

  // Handle "typeN*" or  "type vector[N]*".
  bool IsPtrToVec = TypeStr.consume_back("*");

  if (TypeStr.consume_front(" vector[")) {
    TypeStr = TypeStr.substr(0, TypeStr.find(']'));
  }
  TypeStr.getAsInteger(10, VecElts);
  if (VecElts > 0)
    SpirvTy = getOrCreateSPIRVVectorType(SpirvTy, VecElts, MIRBuilder);

  if (IsPtrToVec)
    SpirvTy = getOrCreateSPIRVPointerType(SpirvTy, MIRBuilder, SC);

  return SpirvTy;
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                                 MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), BitWidth),
      MIRBuilder);
}

SPIRVType *SPIRVGlobalRegistry::finishCreatingSPIRVType(const Type *LLVMTy,
                                                        SPIRVType *SpirvType) {
  assert(CurMF == SpirvType->getMF());
  VRegToTypeMap[CurMF][getSPIRVTypeID(SpirvType)] = SpirvType;
  SPIRVToLLVMType[SpirvType] = LLVMTy;
  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(
    unsigned BitWidth, MachineInstr &I, const SPIRVInstrInfo &TII) {
  Type *LLVMTy = IntegerType::get(CurMF->getFunction().getContext(), BitWidth);
  Register Reg = DT.find(LLVMTy, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeInt))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addImm(BitWidth)
                 .addImm(0);
  DT.add(LLVMTy, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), 1),
      MIRBuilder);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVBoolType(MachineInstr &I,
                                              const SPIRVInstrInfo &TII) {
  Type *LLVMTy = IntegerType::get(CurMF->getFunction().getContext(), 1);
  Register Reg = DT.find(LLVMTy, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeBool))
                 .addDef(createTypeVReg(CurMF->getRegInfo()));
  DT.add(LLVMTy, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVVectorType(
    SPIRVType *BaseType, unsigned NumElements, MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      FixedVectorType::get(const_cast<Type *>(getTypeForSPIRVType(BaseType)),
                           NumElements),
      MIRBuilder);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVVectorType(
    SPIRVType *BaseType, unsigned NumElements, MachineInstr &I,
    const SPIRVInstrInfo &TII) {
  Type *LLVMTy = FixedVectorType::get(
      const_cast<Type *>(getTypeForSPIRVType(BaseType)), NumElements);
  Register Reg = DT.find(LLVMTy, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeVector))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addUse(getSPIRVTypeID(BaseType))
                 .addImm(NumElements);
  DT.add(LLVMTy, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVArrayType(
    SPIRVType *BaseType, unsigned NumElements, MachineInstr &I,
    const SPIRVInstrInfo &TII) {
  Type *LLVMTy = ArrayType::get(
      const_cast<Type *>(getTypeForSPIRVType(BaseType)), NumElements);
  Register Reg = DT.find(LLVMTy, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  MachineBasicBlock &BB = *I.getParent();
  SPIRVType *SpirvType = getOrCreateSPIRVIntegerType(32, I, TII);
  Register Len = getOrCreateConstInt(NumElements, I, SpirvType, TII);
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeArray))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addUse(getSPIRVTypeID(BaseType))
                 .addUse(Len);
  DT.add(LLVMTy, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
    SPIRV::StorageClass::StorageClass SC) {
  const Type *PointerElementType = getTypeForSPIRVType(BaseType);
  unsigned AddressSpace = storageClassToAddressSpace(SC);
  Type *LLVMTy =
      PointerType::get(const_cast<Type *>(PointerElementType), AddressSpace);
  Register Reg = DT.find(PointerElementType, AddressSpace, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  auto MIB = BuildMI(MIRBuilder.getMBB(), MIRBuilder.getInsertPt(),
                     MIRBuilder.getDebugLoc(),
                     MIRBuilder.getTII().get(SPIRV::OpTypePointer))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addImm(static_cast<uint32_t>(SC))
                 .addUse(getSPIRVTypeID(BaseType));
  DT.add(PointerElementType, AddressSpace, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    SPIRVType *BaseType, MachineInstr &I, const SPIRVInstrInfo &TII,
    SPIRV::StorageClass::StorageClass SC) {
  const Type *PointerElementType = getTypeForSPIRVType(BaseType);
  unsigned AddressSpace = storageClassToAddressSpace(SC);
  Type *LLVMTy =
      PointerType::get(const_cast<Type *>(PointerElementType), AddressSpace);
  Register Reg = DT.find(PointerElementType, AddressSpace, CurMF);
  if (Reg.isValid())
    return getSPIRVTypeForVReg(Reg);
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypePointer))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addImm(static_cast<uint32_t>(SC))
                 .addUse(getSPIRVTypeID(BaseType));
  DT.add(PointerElementType, AddressSpace, CurMF, getSPIRVTypeID(MIB));
  return finishCreatingSPIRVType(LLVMTy, MIB);
}

Register SPIRVGlobalRegistry::getOrCreateUndef(MachineInstr &I,
                                               SPIRVType *SpvType,
                                               const SPIRVInstrInfo &TII) {
  assert(SpvType);
  const Type *LLVMTy = getTypeForSPIRVType(SpvType);
  assert(LLVMTy);
  // Find a constant in DT or build a new one.
  UndefValue *UV = UndefValue::get(const_cast<Type *>(LLVMTy));
  Register Res = DT.find(UV, CurMF);
  if (Res.isValid())
    return Res;
  LLT LLTy = LLT::scalar(32);
  Res = CurMF->getRegInfo().createGenericVirtualRegister(LLTy);
  CurMF->getRegInfo().setRegClass(Res, &SPIRV::IDRegClass);
  assignSPIRVTypeToVReg(SpvType, Res, *CurMF);
  DT.add(UV, CurMF, Res);

  MachineInstrBuilder MIB;
  MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(SPIRV::OpUndef))
            .addDef(Res)
            .addUse(getSPIRVTypeID(SpvType));
  const auto &ST = CurMF->getSubtarget();
  constrainSelectedInstRegOperands(*MIB, *ST.getInstrInfo(),
                                   *ST.getRegisterInfo(), *ST.getRegBankInfo());
  return Res;
}
