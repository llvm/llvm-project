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
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"

using namespace llvm;
SPIRVGlobalRegistry::SPIRVGlobalRegistry(unsigned PointerSize)
    : PointerSize(PointerSize) {}

SPIRVType *SPIRVGlobalRegistry::assignTypeToVReg(
    const Type *Type, Register VReg, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier AccessQual, bool EmitIR) {

  SPIRVType *SpirvType =
      getOrCreateSPIRVType(Type, MIRBuilder, AccessQual, EmitIR);
  assignSPIRVTypeToVReg(SpirvType, VReg, MIRBuilder);
  return SpirvType;
}

void SPIRVGlobalRegistry::assignSPIRVTypeToVReg(SPIRVType *SpirvType,
                                                Register VReg,
                                                MachineIRBuilder &MIRBuilder) {
  VRegToTypeMap[&MIRBuilder.getMF()][VReg] = SpirvType;
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

Register SPIRVGlobalRegistry::buildConstantInt(uint64_t Val,
                                               MachineIRBuilder &MIRBuilder,
                                               SPIRVType *SpvType,
                                               bool EmitIR) {
  auto &MF = MIRBuilder.getMF();
  Register Res;
  const IntegerType *LLVMIntTy;
  if (SpvType)
    LLVMIntTy = cast<IntegerType>(getTypeForSPIRVType(SpvType));
  else
    LLVMIntTy = IntegerType::getInt32Ty(MF.getFunction().getContext());
  // Find a constant in DT or build a new one.
  const auto ConstInt =
      ConstantInt::get(const_cast<IntegerType *>(LLVMIntTy), Val);
  unsigned BitWidth = SpvType ? getScalarOrVectorBitWidth(SpvType) : 32;
  Res = MF.getRegInfo().createGenericVirtualRegister(LLT::scalar(BitWidth));
  assignTypeToVReg(LLVMIntTy, Res, MIRBuilder);
  if (EmitIR)
    MIRBuilder.buildConstant(Res, *ConstInt);
  else
    MIRBuilder.buildInstr(SPIRV::OpConstantI)
        .addDef(Res)
        .addImm(ConstInt->getSExtValue());
  return Res;
}

Register SPIRVGlobalRegistry::buildConstantFP(APFloat Val,
                                              MachineIRBuilder &MIRBuilder,
                                              SPIRVType *SpvType) {
  auto &MF = MIRBuilder.getMF();
  Register Res;
  const Type *LLVMFPTy;
  if (SpvType) {
    LLVMFPTy = getTypeForSPIRVType(SpvType);
    assert(LLVMFPTy->isFloatingPointTy());
  } else {
    LLVMFPTy = IntegerType::getFloatTy(MF.getFunction().getContext());
  }
  // Find a constant in DT or build a new one.
  const auto ConstFP = ConstantFP::get(LLVMFPTy->getContext(), Val);
  unsigned BitWidth = SpvType ? getScalarOrVectorBitWidth(SpvType) : 32;
  Res = MF.getRegInfo().createGenericVirtualRegister(LLT::scalar(BitWidth));
  assignTypeToVReg(LLVMFPTy, Res, MIRBuilder);
  MIRBuilder.buildFConstant(Res, *ConstFP);
  return Res;
}

Register SPIRVGlobalRegistry::buildGlobalVariable(
    Register ResVReg, SPIRVType *BaseType, StringRef Name,
    const GlobalValue *GV, SPIRV::StorageClass Storage,
    const MachineInstr *Init, bool IsConst, bool HasLinkageTy,
    SPIRV::LinkageType LinkageType, MachineIRBuilder &MIRBuilder,
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
  Register Reg;
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

  // Set to Reg the same type as ResVReg has.
  auto MRI = MIRBuilder.getMRI();
  assert(MRI->getType(ResVReg).isPointer() && "Pointer type is expected");
  if (Reg != ResVReg) {
    LLT RegLLTy = LLT::pointer(MRI->getType(ResVReg).getAddressSpace(), 32);
    MRI->setType(Reg, RegLLTy);
    assignSPIRVTypeToVReg(BaseType, Reg, MIRBuilder);
  }

  // If it's a global variable with name, output OpName for it.
  if (GVar && GVar->hasName())
    buildOpName(Reg, GVar->getName(), MIRBuilder);

  // Output decorations for the GV.
  // TODO: maybe move to GenerateDecorations pass.
  if (IsConst)
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::Constant, {});

  if (GVar && GVar->getAlign().valueOrOne().value() != 1)
    buildOpDecorate(
        Reg, MIRBuilder, SPIRV::Decoration::Alignment,
        {static_cast<uint32_t>(GVar->getAlign().valueOrOne().value())});

  if (HasLinkageTy)
    buildOpDecorate(Reg, MIRBuilder, SPIRV::Decoration::LinkageAttributes,
                    {static_cast<uint32_t>(LinkageType)}, Name);
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

SPIRVType *SPIRVGlobalRegistry::getOpTypePointer(SPIRV::StorageClass SC,
                                                 SPIRVType *ElemType,
                                                 MachineIRBuilder &MIRBuilder) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpTypePointer)
                 .addDef(createTypeVReg(MIRBuilder))
                 .addImm(static_cast<uint32_t>(SC))
                 .addUse(getSPIRVTypeID(ElemType));
  return MIB;
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

SPIRVType *SPIRVGlobalRegistry::createSPIRVType(const Type *Ty,
                                                MachineIRBuilder &MIRBuilder,
                                                SPIRV::AccessQualifier AccQual,
                                                bool EmitIR) {
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
    auto El = getOrCreateSPIRVType(cast<FixedVectorType>(Ty)->getElementType(),
                                   MIRBuilder);
    return getOpTypeVector(cast<FixedVectorType>(Ty)->getNumElements(), El,
                           MIRBuilder);
  }
  if (Ty->isArrayTy()) {
    auto *El = getOrCreateSPIRVType(Ty->getArrayElementType(), MIRBuilder);
    return getOpTypeArray(Ty->getArrayNumElements(), El, MIRBuilder, EmitIR);
  }
  assert(!isa<StructType>(Ty) && "Unsupported StructType");
  if (auto FType = dyn_cast<FunctionType>(Ty)) {
    SPIRVType *RetTy = getOrCreateSPIRVType(FType->getReturnType(), MIRBuilder);
    SmallVector<SPIRVType *, 4> ParamTypes;
    for (const auto &t : FType->params()) {
      ParamTypes.push_back(getOrCreateSPIRVType(t, MIRBuilder));
    }
    return getOpTypeFunction(RetTy, ParamTypes, MIRBuilder);
  }
  if (auto PType = dyn_cast<PointerType>(Ty)) {
    Type *ElemType = PType->getPointerElementType();

    // Some OpenCL and SPIRV builtins like image2d_t are passed in as pointers,
    // but should be treated as custom types like OpTypeImage.
    assert(!isa<StructType>(ElemType) && "Unsupported StructType pointer");

    // Otherwise, treat it as a regular pointer type.
    auto SC = addressSpaceToStorageClass(PType->getAddressSpace());
    SPIRVType *SpvElementType = getOrCreateSPIRVType(
        ElemType, MIRBuilder, SPIRV::AccessQualifier::ReadWrite, EmitIR);
    return getOpTypePointer(SC, SpvElementType, MIRBuilder);
  }
  llvm_unreachable("Unable to convert LLVM type to SPIRVType");
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
    const Type *Type, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier AccessQual, bool EmitIR) {
  SPIRVType *SpirvType = createSPIRVType(Type, MIRBuilder, AccessQual, EmitIR);
  VRegToTypeMap[&MIRBuilder.getMF()][getSPIRVTypeID(SpirvType)] = SpirvType;
  SPIRVToLLVMType[SpirvType] = Type;
  return SpirvType;
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

SPIRV::StorageClass
SPIRVGlobalRegistry::getPointerStorageClass(Register VReg) const {
  SPIRVType *Type = getSPIRVTypeForVReg(VReg);
  assert(Type && Type->getOpcode() == SPIRV::OpTypePointer &&
         Type->getOperand(1).isImm() && "Pointer type is expected");
  return static_cast<SPIRV::StorageClass>(Type->getOperand(1).getImm());
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                                 MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), BitWidth),
      MIRBuilder);
}

SPIRVType *SPIRVGlobalRegistry::restOfCreateSPIRVType(Type *LLVMTy,
                                                      MachineInstrBuilder MIB) {
  SPIRVType *SpirvType = MIB;
  VRegToTypeMap[CurMF][getSPIRVTypeID(SpirvType)] = SpirvType;
  SPIRVToLLVMType[SpirvType] = LLVMTy;
  return SpirvType;
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVIntegerType(
    unsigned BitWidth, MachineInstr &I, const SPIRVInstrInfo &TII) {
  Type *LLVMTy = IntegerType::get(CurMF->getFunction().getContext(), BitWidth);
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeInt))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addImm(BitWidth)
                 .addImm(0);
  return restOfCreateSPIRVType(LLVMTy, MIB);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder) {
  return getOrCreateSPIRVType(
      IntegerType::get(MIRBuilder.getMF().getFunction().getContext(), 1),
      MIRBuilder);
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
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypeVector))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addUse(getSPIRVTypeID(BaseType))
                 .addImm(NumElements);
  return restOfCreateSPIRVType(LLVMTy, MIB);
}

SPIRVType *
SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(SPIRVType *BaseType,
                                                 MachineIRBuilder &MIRBuilder,
                                                 SPIRV::StorageClass SClass) {
  return getOrCreateSPIRVType(
      PointerType::get(const_cast<Type *>(getTypeForSPIRVType(BaseType)),
                       storageClassToAddressSpace(SClass)),
      MIRBuilder);
}

SPIRVType *SPIRVGlobalRegistry::getOrCreateSPIRVPointerType(
    SPIRVType *BaseType, MachineInstr &I, const SPIRVInstrInfo &TII,
    SPIRV::StorageClass SC) {
  Type *LLVMTy =
      PointerType::get(const_cast<Type *>(getTypeForSPIRVType(BaseType)),
                       storageClassToAddressSpace(SC));
  MachineBasicBlock &BB = *I.getParent();
  auto MIB = BuildMI(BB, I, I.getDebugLoc(), TII.get(SPIRV::OpTypePointer))
                 .addDef(createTypeVReg(CurMF->getRegInfo()))
                 .addImm(static_cast<uint32_t>(SC))
                 .addUse(getSPIRVTypeID(BaseType));
  return restOfCreateSPIRVType(LLVMTy, MIB);
}
