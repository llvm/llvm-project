//===-- SPIRVGlobalRegistry.h - SPIR-V Global Registry ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIRVGlobalRegistry is used to maintain rich type information required for
// SPIR-V even after lowering from LLVM IR to GMIR. It can convert an llvm::Type
// into an OpTypeXXX instruction, and map it to a virtual register. Also it
// builds and supports consistency of constants and global variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRVDuplicatesTracker.h"
#include "SPIRVInstrInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

namespace llvm {
using SPIRVType = const MachineInstr;

class SPIRVGlobalRegistry {
  // Registers holding values which have types associated with them.
  // Initialized upon VReg definition in IRTranslator.
  // Do not confuse this with DuplicatesTracker as DT maps Type* to <MF, Reg>
  // where Reg = OpType...
  // while VRegToTypeMap tracks SPIR-V type assigned to other regs (i.e. not
  // type-declaring ones).
  DenseMap<const MachineFunction *, DenseMap<Register, SPIRVType *>>
      VRegToTypeMap;

  // Map LLVM Type* to <MF, Reg>
  SPIRVGeneralDuplicatesTracker DT;

  DenseMap<SPIRVType *, const Type *> SPIRVToLLVMType;

  // map a Function to its definition (as a machine instruction operand)
  DenseMap<const Function *, const MachineOperand *> FunctionToInstr;
  // map function pointer (as a machine instruction operand) to the used
  // Function
  DenseMap<const MachineOperand *, const Function *> InstrToFunction;

  // Look for an equivalent of the newType in the map. Return the equivalent
  // if it's found, otherwise insert newType to the map and return the type.
  const MachineInstr *checkSpecialInstr(const SPIRV::SpecialTypeDescriptor &TD,
                                        MachineIRBuilder &MIRBuilder);

  SmallPtrSet<const Type *, 4> TypesInProcessing;
  DenseMap<const Type *, SPIRVType *> ForwardPointerTypes;

  // Number of bits pointers and size_t integers require.
  const unsigned PointerSize;

  // Add a new OpTypeXXX instruction without checking for duplicates.
  SPIRVType *createSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                             SPIRV::AccessQualifier::AccessQualifier AQ =
                                 SPIRV::AccessQualifier::ReadWrite,
                             bool EmitIR = true);
  SPIRVType *findSPIRVType(const Type *Ty, MachineIRBuilder &MIRBuilder,
                           SPIRV::AccessQualifier::AccessQualifier accessQual =
                               SPIRV::AccessQualifier::ReadWrite,
                           bool EmitIR = true);
  SPIRVType *
  restOfCreateSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                        SPIRV::AccessQualifier::AccessQualifier AccessQual,
                        bool EmitIR);

public:
  SPIRVGlobalRegistry(unsigned PointerSize);

  MachineFunction *CurMF;

  void add(const Constant *C, MachineFunction *MF, Register R) {
    DT.add(C, MF, R);
  }

  void add(const GlobalVariable *GV, MachineFunction *MF, Register R) {
    DT.add(GV, MF, R);
  }

  void add(const Function *F, MachineFunction *MF, Register R) {
    DT.add(F, MF, R);
  }

  void add(const Argument *Arg, MachineFunction *MF, Register R) {
    DT.add(Arg, MF, R);
  }

  Register find(const Constant *C, MachineFunction *MF) {
    return DT.find(C, MF);
  }

  Register find(const GlobalVariable *GV, MachineFunction *MF) {
    return DT.find(GV, MF);
  }

  Register find(const Function *F, MachineFunction *MF) {
    return DT.find(F, MF);
  }

  void buildDepsGraph(std::vector<SPIRV::DTSortableEntry *> &Graph,
                      MachineModuleInfo *MMI = nullptr) {
    DT.buildDepsGraph(Graph, MMI);
  }

  // Map a machine operand that represents a use of a function via function
  // pointer to a machine operand that represents the function definition.
  // Return either the register or invalid value, because we have no context for
  // a good diagnostic message in case of unexpectedly missing references.
  const MachineOperand *getFunctionDefinitionByUse(const MachineOperand *Use) {
    auto ResF = InstrToFunction.find(Use);
    if (ResF == InstrToFunction.end())
      return nullptr;
    auto ResReg = FunctionToInstr.find(ResF->second);
    return ResReg == FunctionToInstr.end() ? nullptr : ResReg->second;
  }
  // map function pointer (as a machine instruction operand) to the used
  // Function
  void recordFunctionPointer(const MachineOperand *MO, const Function *F) {
    InstrToFunction[MO] = F;
  }
  // map a Function to its definition (as a machine instruction)
  void recordFunctionDefinition(const Function *F, const MachineOperand *MO) {
    FunctionToInstr[F] = MO;
  }
  // Return true if any OpConstantFunctionPointerINTEL were generated
  bool hasConstFunPtr() { return !InstrToFunction.empty(); }

  // Get or create a SPIR-V type corresponding the given LLVM IR type,
  // and map it to the given VReg by creating an ASSIGN_TYPE instruction.
  SPIRVType *assignTypeToVReg(const Type *Type, Register VReg,
                              MachineIRBuilder &MIRBuilder,
                              SPIRV::AccessQualifier::AccessQualifier AQ =
                                  SPIRV::AccessQualifier::ReadWrite,
                              bool EmitIR = true);
  SPIRVType *assignIntTypeToVReg(unsigned BitWidth, Register VReg,
                                 MachineInstr &I, const SPIRVInstrInfo &TII);
  SPIRVType *assignVectTypeToVReg(SPIRVType *BaseType, unsigned NumElements,
                                  Register VReg, MachineInstr &I,
                                  const SPIRVInstrInfo &TII);

  // In cases where the SPIR-V type is already known, this function can be
  // used to map it to the given VReg via an ASSIGN_TYPE instruction.
  void assignSPIRVTypeToVReg(SPIRVType *Type, Register VReg,
                             MachineFunction &MF);

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given LLVM IR type.
  // EmitIR controls if we emit GMIR or SPV constants (e.g. for array sizes)
  // because this method may be called from InstructionSelector and we don't
  // want to emit extra IR instructions there.
  SPIRVType *getOrCreateSPIRVType(const Type *Type,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRV::AccessQualifier::AccessQualifier AQ =
                                      SPIRV::AccessQualifier::ReadWrite,
                                  bool EmitIR = true);

  const Type *getTypeForSPIRVType(const SPIRVType *Ty) const {
    auto Res = SPIRVToLLVMType.find(Ty);
    assert(Res != SPIRVToLLVMType.end());
    return Res->second;
  }

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given string containing the name of the builtin type.
  // Return nullptr if unable to recognize SPIRV type name from `TypeStr`.
  SPIRVType *getOrCreateSPIRVTypeByName(
      StringRef TypeStr, MachineIRBuilder &MIRBuilder,
      SPIRV::StorageClass::StorageClass SC = SPIRV::StorageClass::Function,
      SPIRV::AccessQualifier::AccessQualifier AQ =
          SPIRV::AccessQualifier::ReadWrite);

  // Return the SPIR-V type instruction corresponding to the given VReg, or
  // nullptr if no such type instruction exists.
  SPIRVType *getSPIRVTypeForVReg(Register VReg) const;

  // Whether the given VReg has a SPIR-V type mapped to it yet.
  bool hasSPIRVTypeForVReg(Register VReg) const {
    return getSPIRVTypeForVReg(VReg) != nullptr;
  }

  // Return the VReg holding the result of the given OpTypeXXX instruction.
  Register getSPIRVTypeID(const SPIRVType *SpirvType) const;

  void setCurrentFunc(MachineFunction &MF) { CurMF = &MF; }

  // Whether the given VReg has an OpTypeXXX instruction mapped to it with the
  // given opcode (e.g. OpTypeFloat).
  bool isScalarOfType(Register VReg, unsigned TypeOpcode) const;

  // Return true if the given VReg's assigned SPIR-V type is either a scalar
  // matching the given opcode, or a vector with an element type matching that
  // opcode (e.g. OpTypeBool, or OpTypeVector %x 4, where %x is OpTypeBool).
  bool isScalarOrVectorOfType(Register VReg, unsigned TypeOpcode) const;

  // Return number of elements in a vector if the argument is associated with
  // a vector type. Return 1 for a scalar type, and 0 for a missing type.
  unsigned getScalarOrVectorComponentCount(Register VReg) const;
  unsigned getScalarOrVectorComponentCount(SPIRVType *Type) const;

  // For vectors or scalars of booleans, integers and floats, return the scalar
  // type's bitwidth. Otherwise calls llvm_unreachable().
  unsigned getScalarOrVectorBitWidth(const SPIRVType *Type) const;

  // For vectors or scalars of integers and floats, return total bitwidth of the
  // argument. Otherwise returns 0.
  unsigned getNumScalarOrVectorTotalBitWidth(const SPIRVType *Type) const;

  // Returns either pointer to integer type, that may be a type of vector
  // elements or an original type, or nullptr if the argument is niether
  // an integer scalar, nor an integer vector
  const SPIRVType *retrieveScalarOrVectorIntType(const SPIRVType *Type) const;

  // For integer vectors or scalars, return whether the integers are signed.
  bool isScalarOrVectorSigned(const SPIRVType *Type) const;

  // Gets the storage class of the pointer type assigned to this vreg.
  SPIRV::StorageClass::StorageClass getPointerStorageClass(Register VReg) const;

  // Return the number of bits SPIR-V pointers and size_t variables require.
  unsigned getPointerSize() const { return PointerSize; }

  // Returns true if two types are defined and are compatible in a sense of
  // OpBitcast instruction
  bool isBitcastCompatible(const SPIRVType *Type1,
                           const SPIRVType *Type2) const;

private:
  SPIRVType *getOpTypeBool(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeInt(uint32_t Width, MachineIRBuilder &MIRBuilder,
                          bool IsSigned = false);

  SPIRVType *getOpTypeFloat(uint32_t Width, MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeVoid(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeVector(uint32_t NumElems, SPIRVType *ElemType,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeArray(uint32_t NumElems, SPIRVType *ElemType,
                            MachineIRBuilder &MIRBuilder, bool EmitIR = true);

  SPIRVType *getOpTypeOpaque(const StructType *Ty,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeStruct(const StructType *Ty, MachineIRBuilder &MIRBuilder,
                             bool EmitIR = true);

  SPIRVType *getOpTypePointer(SPIRV::StorageClass::StorageClass SC,
                              SPIRVType *ElemType, MachineIRBuilder &MIRBuilder,
                              Register Reg);

  SPIRVType *getOpTypeForwardPointer(SPIRV::StorageClass::StorageClass SC,
                                     MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeFunction(SPIRVType *RetType,
                               const SmallVectorImpl<SPIRVType *> &ArgTypes,
                               MachineIRBuilder &MIRBuilder);

  SPIRVType *
  getOrCreateSpecialType(const Type *Ty, MachineIRBuilder &MIRBuilder,
                         SPIRV::AccessQualifier::AccessQualifier AccQual);

  std::tuple<Register, ConstantInt *, bool> getOrCreateConstIntReg(
      uint64_t Val, SPIRVType *SpvType, MachineIRBuilder *MIRBuilder,
      MachineInstr *I = nullptr, const SPIRVInstrInfo *TII = nullptr);
  SPIRVType *finishCreatingSPIRVType(const Type *LLVMTy, SPIRVType *SpirvType);
  Register getOrCreateIntCompositeOrNull(uint64_t Val, MachineInstr &I,
                                         SPIRVType *SpvType,
                                         const SPIRVInstrInfo &TII,
                                         Constant *CA, unsigned BitWidth,
                                         unsigned ElemCnt);
  Register getOrCreateIntCompositeOrNull(uint64_t Val,
                                         MachineIRBuilder &MIRBuilder,
                                         SPIRVType *SpvType, bool EmitIR,
                                         Constant *CA, unsigned BitWidth,
                                         unsigned ElemCnt);

public:
  Register buildConstantInt(uint64_t Val, MachineIRBuilder &MIRBuilder,
                            SPIRVType *SpvType = nullptr, bool EmitIR = true);
  Register getOrCreateConstInt(uint64_t Val, MachineInstr &I,
                               SPIRVType *SpvType, const SPIRVInstrInfo &TII);
  Register buildConstantFP(APFloat Val, MachineIRBuilder &MIRBuilder,
                           SPIRVType *SpvType = nullptr);
  Register getOrCreateConsIntVector(uint64_t Val, MachineInstr &I,
                                    SPIRVType *SpvType,
                                    const SPIRVInstrInfo &TII);
  Register getOrCreateConsIntArray(uint64_t Val, MachineInstr &I,
                                   SPIRVType *SpvType,
                                   const SPIRVInstrInfo &TII);
  Register getOrCreateConsIntVector(uint64_t Val, MachineIRBuilder &MIRBuilder,
                                    SPIRVType *SpvType, bool EmitIR = true);
  Register getOrCreateConsIntArray(uint64_t Val, MachineIRBuilder &MIRBuilder,
                                   SPIRVType *SpvType, bool EmitIR = true);
  Register getOrCreateConstNullPtr(MachineIRBuilder &MIRBuilder,
                                   SPIRVType *SpvType);
  Register buildConstantSampler(Register Res, unsigned AddrMode, unsigned Param,
                                unsigned FilerMode,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVType *SpvType);
  Register getOrCreateUndef(MachineInstr &I, SPIRVType *SpvType,
                            const SPIRVInstrInfo &TII);
  Register buildGlobalVariable(Register Reg, SPIRVType *BaseType,
                               StringRef Name, const GlobalValue *GV,
                               SPIRV::StorageClass::StorageClass Storage,
                               const MachineInstr *Init, bool IsConst,
                               bool HasLinkageTy,
                               SPIRV::LinkageType::LinkageType LinkageType,
                               MachineIRBuilder &MIRBuilder,
                               bool IsInstSelector);

  // Convenient helpers for getting types with check for duplicates.
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                         MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth, MachineInstr &I,
                                         const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVBoolType(MachineInstr &I,
                                      const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements,
                                        MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements, MachineInstr &I,
                                        const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVArrayType(SPIRVType *BaseType,
                                       unsigned NumElements, MachineInstr &I,
                                       const SPIRVInstrInfo &TII);

  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
      SPIRV::StorageClass::StorageClass SClass = SPIRV::StorageClass::Function);
  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineInstr &I, const SPIRVInstrInfo &TII,
      SPIRV::StorageClass::StorageClass SClass = SPIRV::StorageClass::Function);

  SPIRVType *
  getOrCreateOpTypeImage(MachineIRBuilder &MIRBuilder, SPIRVType *SampledType,
                         SPIRV::Dim::Dim Dim, uint32_t Depth, uint32_t Arrayed,
                         uint32_t Multisampled, uint32_t Sampled,
                         SPIRV::ImageFormat::ImageFormat ImageFormat,
                         SPIRV::AccessQualifier::AccessQualifier AccQual);

  SPIRVType *getOrCreateOpTypeSampler(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOrCreateOpTypeSampledImage(SPIRVType *ImageType,
                                           MachineIRBuilder &MIRBuilder);

  SPIRVType *
  getOrCreateOpTypePipe(MachineIRBuilder &MIRBuilder,
                        SPIRV::AccessQualifier::AccessQualifier AccQual);
  SPIRVType *getOrCreateOpTypeDeviceEvent(MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateOpTypeFunctionWithArgs(
      const Type *Ty, SPIRVType *RetType,
      const SmallVectorImpl<SPIRVType *> &ArgTypes,
      MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateOpTypeByOpcode(const Type *Ty,
                                       MachineIRBuilder &MIRBuilder,
                                       unsigned Opcode);
};
} // end namespace llvm
#endif // LLLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
