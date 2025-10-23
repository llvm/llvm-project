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
#include "SPIRVIRMapping.h"
#include "SPIRVInstrInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/TypedPointerType.h"

namespace llvm {
class SPIRVSubtarget;
using SPIRVType = const MachineInstr;
using StructOffsetDecorator = std::function<void(Register)>;

class SPIRVGlobalRegistry : public SPIRVIRMapping {
  // Registers holding values which have types associated with them.
  // Initialized upon VReg definition in IRTranslator.
  // Do not confuse this with DuplicatesTracker as DT maps Type* to <MF, Reg>
  // where Reg = OpType...
  // while VRegToTypeMap tracks SPIR-V type assigned to other regs (i.e. not
  // type-declaring ones).
  DenseMap<const MachineFunction *, DenseMap<Register, SPIRVType *>>
      VRegToTypeMap;

  DenseMap<SPIRVType *, const Type *> SPIRVToLLVMType;

  // map a Function to its definition (as a machine instruction operand)
  DenseMap<const Function *, const MachineOperand *> FunctionToInstr;
  DenseMap<const MachineInstr *, const Function *> FunctionToInstrRev;
  // map function pointer (as a machine instruction operand) to the used
  // Function
  DenseMap<const MachineOperand *, const Function *> InstrToFunction;
  // Maps Functions to their calls (in a form of the machine instruction,
  // OpFunctionCall) that happened before the definition is available
  DenseMap<const Function *, SmallPtrSet<MachineInstr *, 8>> ForwardCalls;
  // map a Function to its original return type before the clone function was
  // created during substitution of aggregate arguments
  // (see `SPIRVPrepareFunctions::removeAggregateTypesFromSignature()`)
  DenseMap<Value *, Type *> MutatedAggRet;
  // map an instruction to its value's attributes (type, name)
  DenseMap<MachineInstr *, std::pair<Type *, std::string>> ValueAttrs;

  SmallPtrSet<const Type *, 4> TypesInProcessing;
  DenseMap<const Type *, SPIRVType *> ForwardPointerTypes;

  // Stores for each function the last inserted SPIR-V Type.
  // See: SPIRVGlobalRegistry::createOpType.
  DenseMap<const MachineFunction *, MachineInstr *> LastInsertedTypeMap;

  // if a function returns a pointer, this is to map it into TypedPointerType
  DenseMap<const Function *, TypedPointerType *> FunResPointerTypes;

  // Number of bits pointers and size_t integers require.
  const unsigned PointerSize;

  // Holds the maximum ID we have in the module.
  unsigned Bound;

  // Maps values associated with untyped pointers into deduced element types of
  // untyped pointers.
  DenseMap<Value *, Type *> DeducedElTys;
  // Maps composite values to deduced types where untyped pointers are replaced
  // with typed ones.
  DenseMap<Value *, Type *> DeducedNestedTys;
  // Maps values to "assign type" calls, thus being a registry of created
  // Intrinsic::spv_assign_ptr_type instructions.
  DenseMap<Value *, CallInst *> AssignPtrTypeInstr;

  // Maps OpVariable and OpFunction-related v-regs to its LLVM IR definition.
  DenseMap<std::pair<const MachineFunction *, Register>, const Value *> Reg2GO;

  // map of aliasing decorations to aliasing metadata
  std::unordered_map<const MDNode *, MachineInstr *> AliasInstMDMap;

  // Add a new OpTypeXXX instruction without checking for duplicates.
  SPIRVType *createSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                             SPIRV::AccessQualifier::AccessQualifier AQ,
                             bool ExplicitLayoutRequired, bool EmitIR);
  SPIRVType *findSPIRVType(const Type *Ty, MachineIRBuilder &MIRBuilder,
                           SPIRV::AccessQualifier::AccessQualifier accessQual,
                           bool ExplicitLayoutRequired, bool EmitIR);
  SPIRVType *
  restOfCreateSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                        SPIRV::AccessQualifier::AccessQualifier AccessQual,
                        bool ExplicitLayoutRequired, bool EmitIR);

  // Internal function creating the an OpType at the correct position in the
  // function by tweaking the passed "MIRBuilder" insertion point and restoring
  // it to the correct position. "Op" should be the function creating the
  // specific OpType you need, and should return the newly created instruction.
  SPIRVType *createOpType(MachineIRBuilder &MIRBuilder,
                          std::function<MachineInstr *(MachineIRBuilder &)> Op);

public:
  SPIRVGlobalRegistry(unsigned PointerSize);

  MachineFunction *CurMF;

  void setBound(unsigned V) { Bound = V; }
  unsigned getBound() { return Bound; }

  void addGlobalObject(const Value *V, const MachineFunction *MF, Register R) {
    Reg2GO[std::make_pair(MF, R)] = V;
  }
  const Value *getGlobalObject(const MachineFunction *MF, Register R) {
    auto It = Reg2GO.find(std::make_pair(MF, R));
    return It == Reg2GO.end() ? nullptr : It->second;
  }

  // Add a record to the map of function return pointer types.
  void addReturnType(const Function *ArgF, TypedPointerType *DerivedTy) {
    FunResPointerTypes[ArgF] = DerivedTy;
  }
  // Find a record in the map of function return pointer types.
  const TypedPointerType *findReturnType(const Function *ArgF) {
    auto It = FunResPointerTypes.find(ArgF);
    return It == FunResPointerTypes.end() ? nullptr : It->second;
  }

  // A registry of "assign type" records:
  // - Add a record.
  void addAssignPtrTypeInstr(Value *Val, CallInst *AssignPtrTyCI) {
    AssignPtrTypeInstr[Val] = AssignPtrTyCI;
  }
  // - Find a record.
  CallInst *findAssignPtrTypeInstr(const Value *Val) {
    auto It = AssignPtrTypeInstr.find(Val);
    return It == AssignPtrTypeInstr.end() ? nullptr : It->second;
  }
  // - Find a record and update its key or add a new record, if found.
  void updateIfExistAssignPtrTypeInstr(Value *OldVal, Value *NewVal,
                                       bool DeleteOld) {
    if (CallInst *CI = findAssignPtrTypeInstr(OldVal)) {
      if (DeleteOld)
        AssignPtrTypeInstr.erase(OldVal);
      AssignPtrTypeInstr[NewVal] = CI;
    }
  }

  // A registry of mutated values
  // (see `SPIRVPrepareFunctions::removeAggregateTypesFromSignature()`):
  // - Add a record.
  void addMutated(Value *Val, Type *Ty) { MutatedAggRet[Val] = Ty; }
  // - Find a record.
  Type *findMutated(const Value *Val) {
    auto It = MutatedAggRet.find(Val);
    return It == MutatedAggRet.end() ? nullptr : It->second;
  }

  // A registry of value's attributes (type, name)
  // - Add a record.
  void addValueAttrs(MachineInstr *Key, std::pair<Type *, std::string> Val) {
    ValueAttrs[Key] = Val;
  }
  // - Find a record.
  bool findValueAttrs(const MachineInstr *Key, Type *&Ty, StringRef &Name) {
    auto It = ValueAttrs.find(Key);
    if (It == ValueAttrs.end())
      return false;
    Ty = It->second.first;
    Name = It->second.second;
    return true;
  }

  // Deduced element types of untyped pointers and composites:
  // - Add a record to the map of deduced element types.
  void addDeducedElementType(Value *Val, Type *Ty) { DeducedElTys[Val] = Ty; }
  // - Find a record in the map of deduced element types.
  Type *findDeducedElementType(const Value *Val) {
    auto It = DeducedElTys.find(Val);
    return It == DeducedElTys.end() ? nullptr : It->second;
  }
  // - Find a record and update its key or add a new record, if found.
  void updateIfExistDeducedElementType(Value *OldVal, Value *NewVal,
                                       bool DeleteOld) {
    if (Type *Ty = findDeducedElementType(OldVal)) {
      if (DeleteOld)
        DeducedElTys.erase(OldVal);
      DeducedElTys[NewVal] = Ty;
    }
  }
  // - Add a record to the map of deduced composite types.
  void addDeducedCompositeType(Value *Val, Type *Ty) {
    DeducedNestedTys[Val] = Ty;
  }
  // - Find a record in the map of deduced composite types.
  Type *findDeducedCompositeType(const Value *Val) {
    auto It = DeducedNestedTys.find(Val);
    return It == DeducedNestedTys.end() ? nullptr : It->second;
  }
  // - Find a type of the given Global value
  Type *getDeducedGlobalValueType(const GlobalValue *Global) {
    // we may know element type if it was deduced earlier
    Type *ElementTy = findDeducedElementType(Global);
    if (!ElementTy) {
      // or we may know element type if it's associated with a composite
      // value
      if (Value *GlobalElem =
              Global->getNumOperands() > 0 ? Global->getOperand(0) : nullptr)
        ElementTy = findDeducedCompositeType(GlobalElem);
    }
    return ElementTy ? ElementTy : Global->getValueType();
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

  // Map a Function to a machine instruction that represents the function
  // definition.
  const MachineInstr *getFunctionDefinition(const Function *F) {
    if (!F)
      return nullptr;
    auto MOIt = FunctionToInstr.find(F);
    return MOIt == FunctionToInstr.end() ? nullptr : MOIt->second->getParent();
  }

  // Map a Function to a machine instruction that represents the function
  // definition.
  const Function *getFunctionByDefinition(const MachineInstr *MI) {
    if (!MI)
      return nullptr;
    auto FIt = FunctionToInstrRev.find(MI);
    return FIt == FunctionToInstrRev.end() ? nullptr : FIt->second;
  }

  // map function pointer (as a machine instruction operand) to the used
  // Function
  void recordFunctionPointer(const MachineOperand *MO, const Function *F) {
    InstrToFunction[MO] = F;
  }

  // map a Function to its definition (as a machine instruction)
  void recordFunctionDefinition(const Function *F, const MachineOperand *MO) {
    FunctionToInstr[F] = MO;
    FunctionToInstrRev[MO->getParent()] = F;
  }

  // Return true if any OpConstantFunctionPointerINTEL were generated
  bool hasConstFunPtr() { return !InstrToFunction.empty(); }

  // Add a record about forward function call.
  void addForwardCall(const Function *F, MachineInstr *MI) {
    ForwardCalls[F].insert(MI);
  }

  // Map a Function to the vector of machine instructions that represents
  // forward function calls or to nullptr if not found.
  SmallPtrSet<MachineInstr *, 8> *getForwardCalls(const Function *F) {
    auto It = ForwardCalls.find(F);
    return It == ForwardCalls.end() ? nullptr : &It->second;
  }

  // Get or create a SPIR-V type corresponding the given LLVM IR type,
  // and map it to the given VReg by creating an ASSIGN_TYPE instruction.
  SPIRVType *assignTypeToVReg(const Type *Type, Register VReg,
                              MachineIRBuilder &MIRBuilder,
                              SPIRV::AccessQualifier::AccessQualifier AQ,
                              bool EmitIR);
  SPIRVType *assignIntTypeToVReg(unsigned BitWidth, Register VReg,
                                 MachineInstr &I, const SPIRVInstrInfo &TII);
  SPIRVType *assignFloatTypeToVReg(unsigned BitWidth, Register VReg,
                                   MachineInstr &I, const SPIRVInstrInfo &TII);
  SPIRVType *assignVectTypeToVReg(SPIRVType *BaseType, unsigned NumElements,
                                  Register VReg, MachineInstr &I,
                                  const SPIRVInstrInfo &TII);

  // In cases where the SPIR-V type is already known, this function can be
  // used to map it to the given VReg via an ASSIGN_TYPE instruction.
  void assignSPIRVTypeToVReg(SPIRVType *Type, Register VReg,
                             const MachineFunction &MF);

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given LLVM IR type.
  // EmitIR controls if we emit GMIR or SPV constants (e.g. for array sizes)
  // because this method may be called from InstructionSelector and we don't
  // want to emit extra IR instructions there.
  SPIRVType *getOrCreateSPIRVType(const Type *Type, MachineInstr &I,
                                  SPIRV::AccessQualifier::AccessQualifier AQ,
                                  bool EmitIR) {
    MachineIRBuilder MIRBuilder(I);
    return getOrCreateSPIRVType(Type, MIRBuilder, AQ, EmitIR);
  }

  SPIRVType *getOrCreateSPIRVType(const Type *Type,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRV::AccessQualifier::AccessQualifier AQ,
                                  bool EmitIR) {
    return getOrCreateSPIRVType(Type, MIRBuilder, AQ, false, EmitIR);
  }

  const Type *getTypeForSPIRVType(const SPIRVType *Ty) const {
    auto Res = SPIRVToLLVMType.find(Ty);
    assert(Res != SPIRVToLLVMType.end());
    return Res->second;
  }

  // Return a pointee's type, or nullptr otherwise.
  SPIRVType *getPointeeType(SPIRVType *PtrType);
  // Return a pointee's type op code, or 0 otherwise.
  unsigned getPointeeTypeOp(Register PtrReg);

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given string containing the name of the builtin type.
  // Return nullptr if unable to recognize SPIRV type name from `TypeStr`.
  SPIRVType *getOrCreateSPIRVTypeByName(
      StringRef TypeStr, MachineIRBuilder &MIRBuilder, bool EmitIR,
      SPIRV::StorageClass::StorageClass SC = SPIRV::StorageClass::Function,
      SPIRV::AccessQualifier::AccessQualifier AQ =
          SPIRV::AccessQualifier::ReadWrite);

  // Return the SPIR-V type instruction corresponding to the given VReg, or
  // nullptr if no such type instruction exists. The second argument MF
  // allows to search for the association in a context of the machine functions
  // than the current one, without switching between different "current" machine
  // functions.
  SPIRVType *getSPIRVTypeForVReg(Register VReg,
                                 const MachineFunction *MF = nullptr) const;

  // Return the result type of the instruction defining the register.
  SPIRVType *getResultType(Register VReg, MachineFunction *MF = nullptr);

  // Whether the given VReg has a SPIR-V type mapped to it yet.
  bool hasSPIRVTypeForVReg(Register VReg) const {
    return getSPIRVTypeForVReg(VReg) != nullptr;
  }

  // Return the VReg holding the result of the given OpTypeXXX instruction.
  Register getSPIRVTypeID(const SPIRVType *SpirvType) const;

  // Return previous value of the current machine function
  MachineFunction *setCurrentFunc(MachineFunction &MF) {
    MachineFunction *Ret = CurMF;
    CurMF = &MF;
    return Ret;
  }

  // Return true if the type is an aggregate type.
  bool isAggregateType(SPIRVType *Type) const {
    return Type && (Type->getOpcode() == SPIRV::OpTypeStruct &&
                    Type->getOpcode() == SPIRV::OpTypeArray);
  }

  // Whether the given VReg has an OpTypeXXX instruction mapped to it with the
  // given opcode (e.g. OpTypeFloat).
  bool isScalarOfType(Register VReg, unsigned TypeOpcode) const;

  // Return true if the given VReg's assigned SPIR-V type is either a scalar
  // matching the given opcode, or a vector with an element type matching that
  // opcode (e.g. OpTypeBool, or OpTypeVector %x 4, where %x is OpTypeBool).
  bool isScalarOrVectorOfType(Register VReg, unsigned TypeOpcode) const;

  // Returns true if `Type` is a resource type. This could be an image type
  // or a struct for a buffer decorated with the block decoration.
  bool isResourceType(SPIRVType *Type) const;

  // Return number of elements in a vector if the argument is associated with
  // a vector type. Return 1 for a scalar type, and 0 for a missing type.
  unsigned getScalarOrVectorComponentCount(Register VReg) const;
  unsigned getScalarOrVectorComponentCount(SPIRVType *Type) const;

  // Return the component type in a vector if the argument is associated with
  // a vector type. Returns the argument itself for other types, and nullptr
  // for a missing type.
  SPIRVType *getScalarOrVectorComponentType(Register VReg) const;
  SPIRVType *getScalarOrVectorComponentType(SPIRVType *Type) const;

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
  SPIRV::StorageClass::StorageClass
  getPointerStorageClass(const SPIRVType *Type) const;

  // Return the number of bits SPIR-V pointers and size_t variables require.
  unsigned getPointerSize() const { return PointerSize; }

  // Returns true if two types are defined and are compatible in a sense of
  // OpBitcast instruction
  bool isBitcastCompatible(const SPIRVType *Type1,
                           const SPIRVType *Type2) const;

  // Informs about removal of the machine instruction and invalidates data
  // structures referring this instruction.
  void invalidateMachineInstr(MachineInstr *MI);

private:
  SPIRVType *getOpTypeBool(MachineIRBuilder &MIRBuilder);

  const Type *adjustIntTypeByWidth(const Type *Ty) const;
  unsigned adjustOpTypeIntWidth(unsigned Width) const;

  SPIRVType *getOrCreateSPIRVType(const Type *Type,
                                  MachineIRBuilder &MIRBuilder,
                                  SPIRV::AccessQualifier::AccessQualifier AQ,
                                  bool ExplicitLayoutRequired, bool EmitIR);

  SPIRVType *getOpTypeInt(unsigned Width, MachineIRBuilder &MIRBuilder,
                          bool IsSigned = false);

  SPIRVType *getOpTypeFloat(uint32_t Width, MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeFloat(uint32_t Width, MachineIRBuilder &MIRBuilder,
                            SPIRV::FPEncoding::FPEncoding FPEncode);

  SPIRVType *getOpTypeVoid(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeVector(uint32_t NumElems, SPIRVType *ElemType,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeArray(uint32_t NumElems, SPIRVType *ElemType,
                            MachineIRBuilder &MIRBuilder,
                            bool ExplicitLayoutRequired, bool EmitIR);

  SPIRVType *getOpTypeOpaque(const StructType *Ty,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeStruct(const StructType *Ty, MachineIRBuilder &MIRBuilder,
                             SPIRV::AccessQualifier::AccessQualifier AccQual,
                             StructOffsetDecorator Decorator, bool EmitIR);

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

  SPIRVType *finishCreatingSPIRVType(const Type *LLVMTy, SPIRVType *SpirvType);
  Register getOrCreateBaseRegister(Constant *Val, MachineInstr &I,
                                   SPIRVType *SpvType,
                                   const SPIRVInstrInfo &TII, unsigned BitWidth,
                                   bool ZeroAsNull);
  Register getOrCreateCompositeOrNull(Constant *Val, MachineInstr &I,
                                      SPIRVType *SpvType,
                                      const SPIRVInstrInfo &TII, Constant *CA,
                                      unsigned BitWidth, unsigned ElemCnt,
                                      bool ZeroAsNull = true);

  Register getOrCreateIntCompositeOrNull(uint64_t Val,
                                         MachineIRBuilder &MIRBuilder,
                                         SPIRVType *SpvType, bool EmitIR,
                                         Constant *CA, unsigned BitWidth,
                                         unsigned ElemCnt);

  // Returns a pointer to a SPIR-V pointer type with the given base type and
  // storage class. It is the responsibility of the caller to make sure the
  // decorations on the base type are valid for the given storage class. For
  // example, it has the correct offset and stride decorations.
  SPIRVType *
  getOrCreateSPIRVPointerTypeInternal(SPIRVType *BaseType,
                                      MachineIRBuilder &MIRBuilder,
                                      SPIRV::StorageClass::StorageClass SC);

  void addStructOffsetDecorations(Register Reg, StructType *Ty,
                                  MachineIRBuilder &MIRBuilder);
  void addArrayStrideDecorations(Register Reg, Type *ElementType,
                                 MachineIRBuilder &MIRBuilder);
  bool hasBlockDecoration(SPIRVType *Type) const;

  SPIRVType *
  getOrCreateOpTypeImage(MachineIRBuilder &MIRBuilder, SPIRVType *SampledType,
                         SPIRV::Dim::Dim Dim, uint32_t Depth, uint32_t Arrayed,
                         uint32_t Multisampled, uint32_t Sampled,
                         SPIRV::ImageFormat::ImageFormat ImageFormat,
                         SPIRV::AccessQualifier::AccessQualifier AccQual);

public:
  Register buildConstantInt(uint64_t Val, MachineIRBuilder &MIRBuilder,
                            SPIRVType *SpvType, bool EmitIR,
                            bool ZeroAsNull = true);
  Register getOrCreateConstInt(uint64_t Val, MachineInstr &I,
                               SPIRVType *SpvType, const SPIRVInstrInfo &TII,
                               bool ZeroAsNull = true);
  Register createConstInt(const ConstantInt *CI, MachineInstr &I,
                          SPIRVType *SpvType, const SPIRVInstrInfo &TII,
                          bool ZeroAsNull);
  Register getOrCreateConstFP(APFloat Val, MachineInstr &I, SPIRVType *SpvType,
                              const SPIRVInstrInfo &TII,
                              bool ZeroAsNull = true);
  Register createConstFP(const ConstantFP *CF, MachineInstr &I,
                         SPIRVType *SpvType, const SPIRVInstrInfo &TII,
                         bool ZeroAsNull);
  Register buildConstantFP(APFloat Val, MachineIRBuilder &MIRBuilder,
                           SPIRVType *SpvType = nullptr);

  Register getOrCreateConstVector(uint64_t Val, MachineInstr &I,
                                  SPIRVType *SpvType, const SPIRVInstrInfo &TII,
                                  bool ZeroAsNull = true);
  Register getOrCreateConstVector(APFloat Val, MachineInstr &I,
                                  SPIRVType *SpvType, const SPIRVInstrInfo &TII,
                                  bool ZeroAsNull = true);
  Register getOrCreateConstIntArray(uint64_t Val, size_t Num, MachineInstr &I,
                                    SPIRVType *SpvType,
                                    const SPIRVInstrInfo &TII);
  Register getOrCreateConsIntVector(uint64_t Val, MachineIRBuilder &MIRBuilder,
                                    SPIRVType *SpvType, bool EmitIR);
  Register getOrCreateConstNullPtr(MachineIRBuilder &MIRBuilder,
                                   SPIRVType *SpvType);
  Register buildConstantSampler(Register Res, unsigned AddrMode, unsigned Param,
                                unsigned FilerMode,
                                MachineIRBuilder &MIRBuilder);
  Register getOrCreateUndef(MachineInstr &I, SPIRVType *SpvType,
                            const SPIRVInstrInfo &TII);
  Register buildGlobalVariable(
      Register Reg, SPIRVType *BaseType, StringRef Name, const GlobalValue *GV,
      SPIRV::StorageClass::StorageClass Storage, const MachineInstr *Init,
      bool IsConst,
      const std::optional<SPIRV::LinkageType::LinkageType> &LinkageType,
      MachineIRBuilder &MIRBuilder, bool IsInstSelector);
  Register getOrCreateGlobalVariableWithBinding(const SPIRVType *VarType,
                                                uint32_t Set, uint32_t Binding,
                                                StringRef Name,
                                                MachineIRBuilder &MIRBuilder);

  // Convenient helpers for getting types with check for duplicates.
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                         MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth, MachineInstr &I,
                                         const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVType(unsigned BitWidth, MachineInstr &I,
                                  const SPIRVInstrInfo &TII,
                                  unsigned SPIRVOPcode, Type *LLVMTy);
  SPIRVType *getOrCreateSPIRVFloatType(unsigned BitWidth, MachineInstr &I,
                                       const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder,
                                      bool EmitIR);
  SPIRVType *getOrCreateSPIRVBoolType(MachineInstr &I,
                                      const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements,
                                        MachineIRBuilder &MIRBuilder,
                                        bool EmitIR);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements, MachineInstr &I,
                                        const SPIRVInstrInfo &TII);

  // Returns a pointer to a SPIR-V pointer type with the given base type and
  // storage class. The base type will be translated to a SPIR-V type, and the
  // appropriate layout decorations will be added to the base type.
  SPIRVType *getOrCreateSPIRVPointerType(const Type *BaseType,
                                         MachineIRBuilder &MIRBuilder,
                                         SPIRV::StorageClass::StorageClass SC);
  SPIRVType *getOrCreateSPIRVPointerType(const Type *BaseType, MachineInstr &I,
                                         SPIRV::StorageClass::StorageClass SC);

  // Returns a pointer to a SPIR-V pointer type with the given base type and
  // storage class. It is the responsibility of the caller to make sure the
  // decorations on the base type are valid for the given storage class. For
  // example, it has the correct offset and stride decorations.
  SPIRVType *getOrCreateSPIRVPointerType(SPIRVType *BaseType,
                                         MachineIRBuilder &MIRBuilder,
                                         SPIRV::StorageClass::StorageClass SC);

  // Returns a pointer to a SPIR-V pointer type that is the same as `PtrType`
  // except the stroage class has been changed to `SC`. It is the responsibility
  // of the caller to be sure that the original and new storage class have the
  // same layout requirements.
  SPIRVType *changePointerStorageClass(SPIRVType *PtrType,
                                       SPIRV::StorageClass::StorageClass SC,
                                       MachineInstr &I);

  SPIRVType *getOrCreateVulkanBufferType(MachineIRBuilder &MIRBuilder,
                                         Type *ElemType,
                                         SPIRV::StorageClass::StorageClass SC,
                                         bool IsWritable, bool EmitIr = false);

  SPIRVType *getOrCreateLayoutType(MachineIRBuilder &MIRBuilder,
                                   const TargetExtType *T, bool EmitIr = false);

  SPIRVType *
  getImageType(const TargetExtType *ExtensionType,
               const SPIRV::AccessQualifier::AccessQualifier Qualifier,
               MachineIRBuilder &MIRBuilder);

  SPIRVType *getOrCreateOpTypeSampler(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOrCreateOpTypeSampledImage(SPIRVType *ImageType,
                                           MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateOpTypeCoopMatr(MachineIRBuilder &MIRBuilder,
                                       const TargetExtType *ExtensionType,
                                       const SPIRVType *ElemType,
                                       uint32_t Scope, uint32_t Rows,
                                       uint32_t Columns, uint32_t Use,
                                       bool EmitIR);
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

  SPIRVType *getOrCreateUnknownType(const Type *Ty,
                                    MachineIRBuilder &MIRBuilder,
                                    unsigned Opcode,
                                    const ArrayRef<MCOperand> Operands);

  const TargetRegisterClass *getRegClass(SPIRVType *SpvType) const;
  LLT getRegType(SPIRVType *SpvType) const;

  MachineInstr *getOrAddMemAliasingINTELInst(MachineIRBuilder &MIRBuilder,
                                             const MDNode *AliasingListMD);
  void buildMemAliasingOpDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                                  uint32_t Dec, const MDNode *GVarMD);
  // Replace all uses of a |Old| with |New| updates the global registry type
  // mappings.
  void replaceAllUsesWith(Value *Old, Value *New, bool DeleteOld = true);

  void buildAssignType(IRBuilder<> &B, Type *Ty, Value *Arg);
  void buildAssignPtr(IRBuilder<> &B, Type *ElemTy, Value *Arg);
  void updateAssignType(CallInst *AssignCI, Value *Arg, Value *OfType);
};
} // end namespace llvm
#endif // LLLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
