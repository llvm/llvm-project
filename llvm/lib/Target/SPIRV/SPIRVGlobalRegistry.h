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
  // type-declaring ones)
  DenseMap<const MachineFunction *, DenseMap<Register, SPIRVType *>>
      VRegToTypeMap;

  SPIRVGeneralDuplicatesTracker DT;

  DenseMap<SPIRVType *, const Type *> SPIRVToLLVMType;

  // Number of bits pointers and size_t integers require.
  const unsigned PointerSize;

  // Add a new OpTypeXXX instruction without checking for duplicates.
  SPIRVType *
  createSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                  SPIRV::AccessQualifier AQ = SPIRV::AccessQualifier::ReadWrite,
                  bool EmitIR = true);

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

  // Get or create a SPIR-V type corresponding the given LLVM IR type,
  // and map it to the given VReg by creating an ASSIGN_TYPE instruction.
  SPIRVType *assignTypeToVReg(
      const Type *Type, Register VReg, MachineIRBuilder &MIRBuilder,
      SPIRV::AccessQualifier AQ = SPIRV::AccessQualifier::ReadWrite,
      bool EmitIR = true);

  // In cases where the SPIR-V type is already known, this function can be
  // used to map it to the given VReg via an ASSIGN_TYPE instruction.
  void assignSPIRVTypeToVReg(SPIRVType *Type, Register VReg,
                             MachineFunction &MF);

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given LLVM IR type.
  // EmitIR controls if we emit GMIR or SPV constants (e.g. for array sizes)
  // because this method may be called from InstructionSelector and we don't
  // want to emit extra IR instructions there.
  SPIRVType *getOrCreateSPIRVType(
      const Type *Type, MachineIRBuilder &MIRBuilder,
      SPIRV::AccessQualifier AQ = SPIRV::AccessQualifier::ReadWrite,
      bool EmitIR = true);

  const Type *getTypeForSPIRVType(const SPIRVType *Ty) const {
    auto Res = SPIRVToLLVMType.find(Ty);
    assert(Res != SPIRVToLLVMType.end());
    return Res->second;
  }

  // Return the SPIR-V type instruction corresponding to the given VReg, or
  // nullptr if no such type instruction exists.
  SPIRVType *getSPIRVTypeForVReg(Register VReg) const;

  // Whether the given VReg has a SPIR-V type mapped to it yet.
  bool hasSPIRVTypeForVReg(Register VReg) const {
    return getSPIRVTypeForVReg(VReg) != nullptr;
  }

  // Return the VReg holding the result of the given OpTypeXXX instruction.
  Register getSPIRVTypeID(const SPIRVType *SpirvType) const {
    assert(SpirvType && "Attempting to get type id for nullptr type.");
    return SpirvType->defs().begin()->getReg();
  }

  void setCurrentFunc(MachineFunction &MF) { CurMF = &MF; }

  // Whether the given VReg has an OpTypeXXX instruction mapped to it with the
  // given opcode (e.g. OpTypeFloat).
  bool isScalarOfType(Register VReg, unsigned TypeOpcode) const;

  // Return true if the given VReg's assigned SPIR-V type is either a scalar
  // matching the given opcode, or a vector with an element type matching that
  // opcode (e.g. OpTypeBool, or OpTypeVector %x 4, where %x is OpTypeBool).
  bool isScalarOrVectorOfType(Register VReg, unsigned TypeOpcode) const;

  // For vectors or scalars of ints/floats, return the scalar type's bitwidth.
  unsigned getScalarOrVectorBitWidth(const SPIRVType *Type) const;

  // For integer vectors or scalars, return whether the integers are signed.
  bool isScalarOrVectorSigned(const SPIRVType *Type) const;

  // Gets the storage class of the pointer type assigned to this vreg.
  SPIRV::StorageClass getPointerStorageClass(Register VReg) const;

  // Return the number of bits SPIR-V pointers and size_t variables require.
  unsigned getPointerSize() const { return PointerSize; }

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

  SPIRVType *getOpTypePointer(SPIRV::StorageClass SC, SPIRVType *ElemType,
                              MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeFunction(SPIRVType *RetType,
                               const SmallVectorImpl<SPIRVType *> &ArgTypes,
                               MachineIRBuilder &MIRBuilder);
  SPIRVType *restOfCreateSPIRVType(const Type *LLVMTy, SPIRVType *SpirvType);

public:
  Register buildConstantInt(uint64_t Val, MachineIRBuilder &MIRBuilder,
                            SPIRVType *SpvType = nullptr, bool EmitIR = true);
  Register buildConstantFP(APFloat Val, MachineIRBuilder &MIRBuilder,
                           SPIRVType *SpvType = nullptr);
  Register
  buildGlobalVariable(Register Reg, SPIRVType *BaseType, StringRef Name,
                      const GlobalValue *GV, SPIRV::StorageClass Storage,
                      const MachineInstr *Init, bool IsConst, bool HasLinkageTy,
                      SPIRV::LinkageType LinkageType,
                      MachineIRBuilder &MIRBuilder, bool IsInstSelector);

  // Convenient helpers for getting types with check for duplicates.
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                         MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth, MachineInstr &I,
                                         const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements,
                                        MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements, MachineInstr &I,
                                        const SPIRVInstrInfo &TII);

  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
      SPIRV::StorageClass SClass = SPIRV::StorageClass::Function);
  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineInstr &I, const SPIRVInstrInfo &TII,
      SPIRV::StorageClass SClass = SPIRV::StorageClass::Function);
};
} // end namespace llvm
#endif // LLLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
