//===- LLVMToSPIRVDbgTran.cpp - Converts debug info to SPIR-V ---*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements translation of debug info from LLVM metadata to SPIR-V
//
//===----------------------------------------------------------------------===//
#include "LLVMToSPIRVDbgTran.h"
#include "SPIRVWriter.h"

#include "llvm/IR/DebugInfo.h"

using namespace SPIRV;

// Public interface

/// This function is looking for debug information in the LLVM module
/// and translates it to SPIRV
void LLVMToSPIRVDbgTran::transDebugMetadata() {
  DIF.processModule(*M);
  if (DIF.compile_unit_count() == 0)
    return;

  DICompileUnit *CU = *DIF.compile_units().begin();
  transDbgEntry(CU);

  for (DIImportedEntity *IE : CU->getImportedEntities())
    transDbgEntry(IE);

  for (const DIType *T : DIF.types())
    transDbgEntry(T);

  for (const DIScope *S : DIF.scopes())
    transDbgEntry(S);

  for (const DIGlobalVariableExpression *G : DIF.global_variables()) {
    transDbgEntry(G->getVariable());
  }

  for (const DISubprogram *F : DIF.subprograms())
    transDbgEntry(F);

  for (const DbgDeclareInst *DDI : DbgDeclareIntrinsics)
    finalizeDebugDeclare(DDI);

  for (const DbgValueInst *DVI : DbgValueIntrinsics)
    finalizeDebugValue(DVI);

  transLocationInfo();
}

// llvm.dbg.declare intrinsic.

SPIRVValue *
LLVMToSPIRVDbgTran::createDebugDeclarePlaceholder(const DbgDeclareInst *DbgDecl,
                                                  SPIRVBasicBlock *BB) {
  if (!DbgDecl->getAddress())
    return nullptr; // The variable is dead.

  DbgDeclareIntrinsics.push_back(DbgDecl);
  using namespace SPIRVDebug::Operand::DebugDeclare;
  SPIRVWordVec Ops(OperandCount, getDebugInfoNone()->getId());
  SPIRVId ExtSetId = BM->getExtInstSetId(SPIRVEIS_Debug);
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Declare, Ops, BB);
}

void LLVMToSPIRVDbgTran::finalizeDebugDeclare(const DbgDeclareInst *DbgDecl) {
  SPIRVValue *V = SPIRVWriter->getTranslatedValue(DbgDecl);
  assert(V && "llvm.dbg.declare intrinsic isn't mapped to a SPIRV instruction");
  assert(V->isExtInst(SPIRV::SPIRVEIS_Debug, SPIRVDebug::Declare) &&
         "llvm.dbg.declare intrinsic has been translated wrong!");
  if (!V || !V->isExtInst(SPIRV::SPIRVEIS_Debug, SPIRVDebug::Declare))
    return;
  SPIRVExtInst *DD = static_cast<SPIRVExtInst *>(V);
  SPIRVBasicBlock *BB = DD->getBasicBlock();
  llvm::Value *Alloca = DbgDecl->getAddress();

  using namespace SPIRVDebug::Operand::DebugDeclare;
  SPIRVWordVec Ops(OperandCount);
  Ops[DebugLocalVarIdx] = transDbgEntry(DbgDecl->getVariable())->getId();
  Ops[VariableIdx] = SPIRVWriter->transValue(Alloca, BB)->getId();
  Ops[ExpressionIdx] = transDbgEntry(DbgDecl->getExpression())->getId();
  DD->setArguments(Ops);
}

// llvm.dbg.value intrinsic.

SPIRVValue *
LLVMToSPIRVDbgTran::createDebugValuePlaceholder(const DbgValueInst *DbgValue,
                                                SPIRVBasicBlock *BB) {
  if (!DbgValue->getValue())
    return nullptr; // It is pointless without new value

  DbgValueIntrinsics.push_back(DbgValue);
  using namespace SPIRVDebug::Operand::DebugValue;
  SPIRVWordVec Ops(MinOperandCount, getDebugInfoNone()->getId());
  SPIRVId ExtSetId = BM->getExtInstSetId(SPIRVEIS_Debug);
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Value, Ops, BB);
}

void LLVMToSPIRVDbgTran::finalizeDebugValue(const DbgValueInst *DbgValue) {
  SPIRVValue *V = SPIRVWriter->getTranslatedValue(DbgValue);
  assert(V && "llvm.dbg.value intrinsic isn't mapped to a SPIRV instruction");
  assert(V->isExtInst(SPIRV::SPIRVEIS_Debug, SPIRVDebug::Value) &&
         "llvm.dbg.value intrinsic has been translated wrong!");
  if (!V || !V->isExtInst(SPIRV::SPIRVEIS_Debug, SPIRVDebug::Value))
    return;
  SPIRVExtInst *DV = static_cast<SPIRVExtInst *>(V);
  SPIRVBasicBlock *BB = DV->getBasicBlock();
  Value *Val = DbgValue->getValue();

  using namespace SPIRVDebug::Operand::DebugValue;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[DebugLocalVarIdx] = transDbgEntry(DbgValue->getVariable())->getId();
  Ops[ValueIdx] = SPIRVWriter->transValue(Val, BB)->getId();
  Ops[ExpressionIdx] = transDbgEntry(DbgValue->getExpression())->getId();
  DV->setArguments(Ops);
}

// Emitting DebugScope and OpLine instructions

void LLVMToSPIRVDbgTran::transLocationInfo() {
  for (const Function &F : *M) {
    for (const BasicBlock &BB : F) {
      SPIRVValue *V = SPIRVWriter->getTranslatedValue(&BB);
      assert(V && V->isBasicBlock() &&
             "Basic block is expected to be translated");
      SPIRVBasicBlock *SBB = static_cast<SPIRVBasicBlock *>(V);
      MDNode *DbgScope = nullptr;
      MDNode *InlinedAt = nullptr;
      SPIRVString *File = nullptr;
      unsigned LineNo = 0;
      unsigned Col = 0;
      for (const Instruction &I : BB) {
        const DebugLoc &DL = I.getDebugLoc();
        if (!DL.get()) {
          if (DbgScope || InlinedAt) { // Emit DebugNoScope
            DbgScope = nullptr;
            InlinedAt = nullptr;
            V = SPIRVWriter->getTranslatedValue(&I);
            transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
          }
          continue;
        }
        // Once scope or inlining has changed emit another DebugScope
        if (DL.getScope() != DbgScope || DL.getInlinedAt() != InlinedAt) {
          DbgScope = DL.getScope();
          InlinedAt = DL.getInlinedAt();
          V = SPIRVWriter->getTranslatedValue(&I);
          transDebugLoc(DL, SBB, static_cast<SPIRVInstruction *>(V));
        }
        // If any component of OpLine has changed emit another OpLine
        SPIRVString *DirAndFile = BM->getString(getFullPath(DL.get()));
        if (File != DirAndFile || LineNo != DL.getLine() ||
            Col != DL.getCol()) {
          File = DirAndFile;
          LineNo = DL.getLine();
          Col = DL.getCol();
          V = SPIRVWriter->getTranslatedValue(&I);
          BM->addLine(V, File ? File->getId() : getDebugInfoNone()->getId(),
                      LineNo, Col);
        }
      } // Instructions
    }   // Basic Blocks
  }     // Functions
}

// Translation of single debug entry

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntry(const MDNode *DIEntry) {
  // Caching
  auto It = MDMap.find(DIEntry);
  if (It != MDMap.end()) {
    assert(It->second && "Invalid SPIRVEntry is cached!");
    return It->second;
  }
  SPIRVEntry *Res = transDbgEntryImpl(DIEntry);
  assert(Res && "Translation failure");
  MDMap[DIEntry] = Res;
  return Res;
}

template <typename T>
SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntryRef(const TypedDINodeRef<T> &Ref,
                                                 SPIRVEntry *Alternate) {
  T *Resolved = Ref.resolve();
  if (!Resolved && Alternate)
    return Alternate;
  return transDbgEntry(Resolved);
}

// Dispatcher implementation

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEntryImpl(const MDNode *MDN) {
  if (!MDN)
    return BM->addDebugInfo(SPIRVDebug::DebugInfoNone, getVoidTy(),
                            SPIRVWordVec());
  if (const DINode *DIEntry = dyn_cast<DINode>(MDN)) {
    switch (DIEntry->getTag()) {
    // Types
    case dwarf::DW_TAG_base_type:
    case dwarf::DW_TAG_unspecified_type:
      return transDbgBaseType(cast<DIBasicType>(DIEntry));

    case dwarf::DW_TAG_reference_type:
    case dwarf::DW_TAG_rvalue_reference_type:
    case dwarf::DW_TAG_pointer_type:
      return transDbgPointerType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_array_type:
      return transDbgArrayType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_volatile_type:
    case dwarf::DW_TAG_atomic_type:
      return transDbgQualifiedType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_subroutine_type:
      return transDbgSubroutineType(cast<DISubroutineType>(DIEntry));

    case dwarf::DW_TAG_class_type:
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_union_type:
      return transDbgCompositeType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_member:
      return transDbgMemberType(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_inheritance:
      return transDbgInheritance(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_enumeration_type:
      return transDbgEnumType(cast<DICompositeType>(DIEntry));

    case dwarf::DW_TAG_file_type:
      return transDbgFileType(cast<DIFile>(DIEntry));

    case dwarf::DW_TAG_typedef:
      return transDbgTypeDef(cast<DIDerivedType>(DIEntry));

    case dwarf::DW_TAG_ptr_to_member_type:
      return transDbgPtrToMember(cast<DIDerivedType>(DIEntry));

    // Scope
    case dwarf::DW_TAG_namespace:
    case dwarf::DW_TAG_lexical_block:
      return transDbgScope(cast<DIScope>(DIEntry));

    // Function
    case dwarf::DW_TAG_subprogram:
      return transDbgFunction(cast<DISubprogram>(DIEntry));

    // Variables
    case dwarf::DW_TAG_variable:
      if (const DILocalVariable *LV = dyn_cast<DILocalVariable>(DIEntry))
        return transDbgLocalVariable(LV);
      if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DIEntry))
        return transDbgGlobalVariable(GV);
      llvm_unreachable("Unxpected debug info type for variable");
    case dwarf::DW_TAG_formal_parameter:
      return transDbgLocalVariable(cast<DILocalVariable>(DIEntry));

    // Compilation unit
    case dwarf::DW_TAG_compile_unit:
      return transDbgCompilationUnit(cast<DICompileUnit>(DIEntry));

    // Templates
    case dwarf::DW_TAG_template_type_parameter:
    case dwarf::DW_TAG_template_value_parameter:
      return transDbgTemplateParameter(cast<DITemplateParameter>(DIEntry));
    case dwarf::DW_TAG_GNU_template_template_param:
      return transDbgTemplateTemplateParameter(
          cast<DITemplateValueParameter>(DIEntry));
    case dwarf::DW_TAG_GNU_template_parameter_pack:
      return transDbgTemplateParameterPack(
          cast<DITemplateValueParameter>(DIEntry));

    case dwarf::DW_TAG_imported_module:
    case dwarf::DW_TAG_imported_declaration:
      return transDbgImportedEntry(cast<DIImportedEntity>(DIEntry));

    default:
      return getDebugInfoNone();
    }
  }
  if (const DIExpression *Expr = dyn_cast<DIExpression>(MDN))
    return transDbgExpression(Expr);

  if (const DILocation *Loc = dyn_cast<DILocation>(MDN)) {
    return transDbgInlinedAt(Loc);
  }
  llvm_unreachable("Not implemented debug info entry!");
}

// Helper methods

SPIRVType *LLVMToSPIRVDbgTran::getVoidTy() {
  if (!VoidT) {
    assert(M && "Pointer to LLVM Module is expected to be initialized!");
    // Cache void type in a member.
    VoidT = SPIRVWriter->transType(Type::getVoidTy(M->getContext()));
  }
  return VoidT;
}

SPIRVEntry *LLVMToSPIRVDbgTran::getScope(DIScopeRef SR) {
  if (DIScope *S = SR.resolve())
    return transDbgEntry(S);
  else {
    assert(SPIRVCU && "Compilation unit must already be translated!");
    return SPIRVCU;
  }
}

SPIRVEntry *LLVMToSPIRVDbgTran::getScope(DIScope *S) {
  if (S)
    return transDbgEntry(S);
  else {
    assert(SPIRVCU && "Compile unit is expected to be already translated");
    return SPIRVCU;
  }
}

SPIRVEntry *LLVMToSPIRVDbgTran::getGlobalVariable(const DIGlobalVariable *GV) {
  for (GlobalVariable &V : M->globals()) {
    SmallVector<DIGlobalVariableExpression *, 4> GVs;
    V.getDebugInfo(GVs);
    for (DIGlobalVariableExpression *GVE : GVs) {
      if (GVE->getVariable() == GV)
        return SPIRVWriter->transValue(&V, nullptr);
    }
  }
  return getDebugInfoNone();
}

SPIRVWord mapDebugFlags(DINode::DIFlags DFlags) {
  SPIRVWord Flags = 0;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPublic)
    Flags |= SPIRVDebug::FlagIsPublic;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagProtected)
    Flags |= SPIRVDebug::FlagIsProtected;
  if ((DFlags & DINode::FlagAccessibility) == DINode::FlagPrivate)
    Flags |= SPIRVDebug::FlagIsPrivate;

  if (DFlags & DINode::FlagFwdDecl)
    Flags |= SPIRVDebug::FlagIsFwdDecl;
  if (DFlags & DINode::FlagArtificial)
    Flags |= SPIRVDebug::FlagIsArtificial;
  if (DFlags & DINode::FlagExplicit)
    Flags |= SPIRVDebug::FlagIsExplicit;
  if (DFlags & DINode::FlagPrototyped)
    Flags |= SPIRVDebug::FlagIsPrototyped;
  if (DFlags & DINode::FlagObjectPointer)
    Flags |= SPIRVDebug::FlagIsObjectPointer;
  if (DFlags & DINode::FlagStaticMember)
    Flags |= SPIRVDebug::FlagIsStaticMember;
  // inderect variable flag ?
  if (DFlags & DINode::FlagLValueReference)
    Flags |= SPIRVDebug::FlagIsLValueReference;
  if (DFlags & DINode::FlagRValueReference)
    Flags |= SPIRVDebug::FlagIsRValueReference;
  return Flags;
}

SPIRVWord transDebugFlags(const DINode *DN) {
  SPIRVWord Flags = 0;
  if (const DIGlobalVariable *GV = dyn_cast<DIGlobalVariable>(DN)) {
    if (GV->isLocalToUnit())
      Flags |= SPIRVDebug::FlagIsLocal;
    if (GV->isDefinition())
      Flags |= SPIRVDebug::FlagIsDefinition;
  }
  if (const DISubprogram *DS = dyn_cast<DISubprogram>(DN)) {
    if (DS->isLocalToUnit())
      Flags |= SPIRVDebug::FlagIsLocal;
    if (DS->isOptimized())
      Flags |= SPIRVDebug::FlagIsOptimized;
    if (DS->isDefinition())
      Flags |= SPIRVDebug::FlagIsDefinition;
    Flags |= mapDebugFlags(DS->getFlags());
  }
  if (DN->getTag() == dwarf::DW_TAG_reference_type)
    Flags |= SPIRVDebug::FlagIsLValueReference;
  if (DN->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    Flags |= SPIRVDebug::FlagIsRValueReference;
  if (const DIType *DT = dyn_cast<DIType>(DN))
    Flags |= mapDebugFlags(DT->getFlags());
  if (const DILocalVariable *DLocVar = dyn_cast<DILocalVariable>(DN))
    Flags |= mapDebugFlags(DLocVar->getFlags());

  return Flags;
}

/// The following methods (till the end of the file) implement translation of
/// debug instrtuctions described in the spec.

// Absent Debug Info

SPIRVEntry *LLVMToSPIRVDbgTran::getDebugInfoNone() {
  if (!DebugInfoNone) {
    DebugInfoNone = transDbgEntry(nullptr);
  }
  return DebugInfoNone;
}

SPIRVId LLVMToSPIRVDbgTran::getDebugInfoNoneId() {
  return getDebugInfoNone()->getId();
}

// Compilation unit

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgCompilationUnit(const DICompileUnit *CU) {
  using namespace SPIRVDebug::Operand::CompilationUnit;
  SPIRVWordVec Ops(OperandCount);
  Ops[SPIRVDebugInfoVersionIdx] = SPIRVDebug::DebugInfoVersion;
  Ops[DWARFVersionIdx] = M->getDwarfVersion();
  Ops[SourceIdx] = getSource(CU)->getId();
  Ops[LanguageIdx] = CU->getSourceLanguage();
  // Cache CU in a member.
  SPIRVCU = static_cast<SPIRVExtInst *>(
      BM->addDebugInfo(SPIRVDebug::CompilationUnit, getVoidTy(), Ops));
  return SPIRVCU;
}

// Types

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgBaseType(const DIBasicType *BT) {
  using namespace SPIRVDebug::Operand::TypeBasic;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(BT->getName())->getId();
  ConstantInt *Size = getUInt(M, BT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  auto Encoding = static_cast<dwarf::TypeKind>(BT->getEncoding());
  SPIRVDebug::EncodingTag EncTag = SPIRVDebug::Unspecified;
  SPIRV::DbgEncodingMap::find(Encoding, &EncTag);
  Ops[EncodingIdx] = EncTag;
  return BM->addDebugInfo(SPIRVDebug::TypeBasic, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgPointerType(const DIDerivedType *PT) {
  using namespace SPIRVDebug::Operand::TypePointer;
  SPIRVWordVec Ops(OperandCount);
  SPIRVEntry *Base = transDbgEntryRef(PT->getBaseType(), getVoidTy());
  Ops[BaseTypeIdx] = Base->getId();
  Ops[StorageClassIdx] = ~0U; // all ones denote no address space
  Optional<unsigned> AS = PT->getDWARFAddressSpace();
  if (AS.hasValue()) {
    SPIRAddressSpace SPIRAS = static_cast<SPIRAddressSpace>(AS.getValue());
    Ops[StorageClassIdx] = SPIRSPIRVAddrSpaceMap::map(SPIRAS);
  }
  Ops[FlagsIdx] = transDebugFlags(PT);
  SPIRVEntry *Res = BM->addDebugInfo(SPIRVDebug::TypePointer, getVoidTy(), Ops);
  return Res;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgQualifiedType(const DIDerivedType *QT) {
  using namespace SPIRVDebug::Operand::TypeQualifier;
  SPIRVWordVec Ops(OperandCount);
  SPIRVEntry *Base = transDbgEntryRef(QT->getBaseType());
  Ops[BaseTypeIdx] = Base->getId();
  Ops[QualifierIdx] = SPIRV::DbgTypeQulifierMap::map(
      static_cast<llvm::dwarf::Tag>(QT->getTag()));
  return BM->addDebugInfo(SPIRVDebug::TypeQualifier, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgArrayType(const DICompositeType *AT) {
  using namespace SPIRVDebug::Operand::TypeArray;
  SPIRVWordVec Ops(MinOperandCount);
  SPIRVEntry *Base = transDbgEntryRef(AT->getBaseType());
  Ops[BaseTypeIdx] = Base->getId();

  DINodeArray AR(AT->getElements());
  // For N-dimensianal arrays AR.getNumElements() == N
  const unsigned N = AR.size();
  Ops.resize(ComponentCountIdx + N);
  for (unsigned I = 0; I < N; ++I) {
    DISubrange *SR = cast<DISubrange>(AR[I]);
    ConstantInt *Count = SR->getCount().get<ConstantInt *>();
    if (AT->isVector()) {
      assert(N == 1 && "Multidimensional vector is not expected!");
      Ops[ComponentCountIdx] = static_cast<SPIRVWord>(Count->getZExtValue());
      return BM->addDebugInfo(SPIRVDebug::TypeVector, getVoidTy(), Ops);
    }
    SPIRVValue *C = SPIRVWriter->transValue(Count, nullptr);
    Ops[ComponentCountIdx + I] = C->getId();
  }
  return BM->addDebugInfo(SPIRVDebug::TypeArray, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTypeDef(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::Typedef;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(DT->getName())->getId();
  SPIRVEntry *BaseTy = transDbgEntryRef(DT->getBaseType());
  assert(BaseTy && "Couldn't translate base type!");
  Ops[BaseTypeIdx] = BaseTy->getId();
  Ops[SourceIdx] = getSource(DT)->getId();
  Ops[LineIdx] = 0;   // This version of DIDerivedType has no line number
  Ops[ColumnIdx] = 0; // This version of DIDerivedType has no column number
  SPIRVEntry *Scope = getScope(DT->getScope());
  assert(Scope && "Couldn't translate scope!");
  Ops[ParentIdx] = Scope->getId();
  return BM->addDebugInfo(SPIRVDebug::Typedef, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgSubroutineType(const DISubroutineType *FT) {
  using namespace SPIRVDebug::Operand::TypeFunction;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[FlagsIdx] = transDebugFlags(FT);

  DITypeRefArray Types = FT->getTypeArray();
  const size_t NumElements = Types.size();
  if (NumElements) {
    Ops.resize(1 + NumElements);
    // First element of the TypeArray is the type of the return value,
    // followed by types of the function arguments' types.
    // The same order is preserved in SPIRV.
    for (unsigned I = 0; I < NumElements; ++I)
      Ops[ReturnTypeIdx + I] = transDbgEntryRef(Types[I], getVoidTy())->getId();
  } else { // void foo();
    Ops[ReturnTypeIdx] = getVoidTy()->getId();
  }

  return BM->addDebugInfo(SPIRVDebug::TypeFunction, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgEnumType(const DICompositeType *ET) {
  using namespace SPIRVDebug::Operand::TypeEnum;
  SPIRVWordVec Ops(MinOperandCount);

  SPIRVEntry *UnderlyingType = getVoidTy();
  if (DITypeRef DerivedFrom = ET->getBaseType())
    UnderlyingType = transDbgEntryRef(DerivedFrom);
  ConstantInt *Size = getUInt(M, ET->getSizeInBits());

  Ops[NameIdx] = BM->getString(ET->getName())->getId();
  Ops[UnderlyingTypeIdx] = UnderlyingType->getId();
  Ops[SourceIdx] = getSource(ET)->getId();
  Ops[LineIdx] = ET->getLine();
  Ops[ColumnIdx] = 0; // This version of DICompositeType has no column number
  Ops[ParentIdx] = getScope(ET->getScope())->getId();
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(ET);

  DINodeArray Elements = ET->getElements();
  size_t ElemCount = Elements.size();
  for (unsigned I = 0; I < ElemCount; ++I) {
    DIEnumerator *E = cast<DIEnumerator>(Elements[I]);
    ConstantInt *EnumValue = getInt(M, E->getValue());
    SPIRVValue *Val = SPIRVWriter->transValue(EnumValue, nullptr);
    assert(Val->getOpCode() == OpConstant &&
           "LLVM constant must be translated to SPIRV constant");
    Ops.push_back(Val->getId());
    SPIRVString *Name = BM->getString(E->getName());
    Ops.push_back(Name->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::TypeEnum, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgCompositeType(const DICompositeType *CT) {
  using namespace SPIRVDebug::Operand::TypeComposite;
  SPIRVWordVec Ops(MinOperandCount);

  SPIRVForward *Tmp = BM->addForward(nullptr);
  MDMap.insert(std::make_pair(CT, Tmp));

  auto Tag = static_cast<dwarf::Tag>(CT->getTag());
  SPIRVId UniqId = getDebugInfoNoneId();
  StringRef Identifier = CT->getIdentifier();
  if (!Identifier.empty())
    UniqId = BM->getString(Identifier)->getId();
  ConstantInt *Size = getUInt(M, CT->getSizeInBits());

  Ops[NameIdx] = BM->getString(CT->getName())->getId();
  Ops[TagIdx] = SPIRV::DbgCompositeTypeMap::map(Tag);
  Ops[SourceIdx] = getSource(CT)->getId();
  Ops[LineIdx] = CT->getLine();
  Ops[ColumnIdx] = 0; // This version of DICompositeType has no column number
  Ops[ParentIdx] = getScope(CT->getScope())->getId();
  Ops[LinkageNameIdx] = UniqId;
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(CT);

  for (DINode *N : CT->getElements()) {
    Ops.push_back(transDbgEntry(N)->getId());
  }

  SPIRVEntry *Res =
      BM->addDebugInfo(SPIRVDebug::TypeComposite, getVoidTy(), Ops);
  BM->replaceForward(Tmp, Res);

  // Translate template parameters.
  if (DITemplateParameterArray TP = CT->getTemplateParams()) {
    const unsigned int NumTParams = TP.size();
    SPIRVWordVec Args(1 + NumTParams);
    Args[0] = Res->getId();
    for (unsigned int I = 0; I < NumTParams; ++I) {
      Args[I + 1] = transDbgEntry(TP[I])->getId();
    }
    Res = BM->addDebugInfo(SPIRVDebug::TypeTemplate, getVoidTy(), Args);
  }
  MDMap[CT] = Res;
  return Res;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgMemberType(const DIDerivedType *MT) {
  using namespace SPIRVDebug::Operand::TypeMember;
  SPIRVWordVec Ops(MinOperandCount);

  Ops[NameIdx] = BM->getString(MT->getName())->getId();
  Ops[TypeIdx] = transDbgEntryRef(MT->getBaseType())->getId();
  Ops[SourceIdx] = getSource(MT)->getId();
  Ops[LineIdx] = MT->getLine();
  Ops[ColumnIdx] = 0; // This version of DIDerivedType has no column number
  Ops[ParentIdx] = transDbgEntryRef(MT->getScope())->getId();
  ConstantInt *Offset = getUInt(M, MT->getOffsetInBits());
  Ops[OffsetIdx] = SPIRVWriter->transValue(Offset, nullptr)->getId();
  ConstantInt *Size = getUInt(M, MT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(MT);
  if (MT->isStaticMember()) {
    if (llvm::Constant *C = MT->getConstant()) {
      SPIRVValue *Val = SPIRVWriter->transValue(C, nullptr);
      assert(isConstantOpCode(Val->getOpCode()) &&
             "LLVM constant must be translated to SPIRV constant");
      Ops.push_back(Val->getId());
    }
  }
  return BM->addDebugInfo(SPIRVDebug::TypeMember, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgInheritance(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::TypeInheritance;
  SPIRVWordVec Ops(OperandCount);
  Ops[ChildIdx] = transDbgEntryRef(DT->getScope())->getId();
  Ops[ParentIdx] = transDbgEntryRef(DT->getBaseType())->getId();
  ConstantInt *Offset = getUInt(M, DT->getOffsetInBits());
  Ops[OffsetIdx] = SPIRVWriter->transValue(Offset, nullptr)->getId();
  ConstantInt *Size = getUInt(M, DT->getSizeInBits());
  Ops[SizeIdx] = SPIRVWriter->transValue(Size, nullptr)->getId();
  Ops[FlagsIdx] = transDebugFlags(DT);
  return BM->addDebugInfo(SPIRVDebug::Inheritance, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgPtrToMember(const DIDerivedType *DT) {
  using namespace SPIRVDebug::Operand::PtrToMember;
  SPIRVWordVec Ops(OperandCount);
  Ops[MemberTypeIdx] = transDbgEntryRef(DT->getBaseType())->getId();
  Ops[ParentIdx] = transDbgEntryRef(DT->getClassType())->getId();
  return BM->addDebugInfo(SPIRVDebug::TypePtrToMember, getVoidTy(), Ops);
}

// Templates
SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgTemplateParams(DITemplateParameterArray TPA,
                                           const SPIRVEntry *Target) {
  using namespace SPIRVDebug::Operand::Template;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[TargetIdx] = Target->getId();
  for (DITemplateParameter *TP : TPA) {
    Ops.push_back(transDbgEntry(TP)->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::TypeTemplate, getVoidTy(), Ops);
}

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgTemplateParameter(const DITemplateParameter *TP) {
  using namespace SPIRVDebug::Operand::TemplateParameter;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(TP->getName())->getId();
  Ops[TypeIdx] = transDbgEntryRef(TP->getType(), getVoidTy())->getId();
  Ops[ValueIdx] = getDebugInfoNoneId();
  if (TP->getTag() == dwarf::DW_TAG_template_value_parameter) {
    const DITemplateValueParameter *TVP = cast<DITemplateValueParameter>(TP);
    Constant *C = cast<ConstantAsMetadata>(TVP->getValue())->getValue();
    Ops[ValueIdx] = SPIRVWriter->transValue(C, nullptr)->getId();
  }
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0;   // This version of DITemplateParameter has no line number
  Ops[ColumnIdx] = 0; // This version of DITemplateParameter has no column info
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateParameter, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTemplateTemplateParameter(
    const DITemplateValueParameter *TVP) {
  using namespace SPIRVDebug::Operand::TemplateTemplateParameter;
  SPIRVWordVec Ops(OperandCount);
  assert(isa<MDString>(TVP->getValue()));
  MDString *Val = cast<MDString>(TVP->getValue());
  Ops[NameIdx] = BM->getString(TVP->getName())->getId();
  Ops[TemplateNameIdx] = BM->getString(Val->getString())->getId();
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0; // This version of DITemplateValueParameter has no line info
  Ops[ColumnIdx] = 0; // This version of DITemplateValueParameter has no column
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateTemplateParameter,
                          getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgTemplateParameterPack(
    const DITemplateValueParameter *TVP) {
  using namespace SPIRVDebug::Operand::TemplateParameterPack;
  SPIRVWordVec Ops(MinOperandCount);
  assert(isa<MDNode>(TVP->getValue()));
  MDNode *Params = cast<MDNode>(TVP->getValue());

  Ops[NameIdx] = BM->getString(TVP->getName())->getId();
  Ops[SourceIdx] = getDebugInfoNoneId();
  Ops[LineIdx] = 0; // This version of DITemplateValueParameter has no line info
  Ops[ColumnIdx] = 0; // This version of DITemplateValueParameter has no column

  for (const MDOperand &Op : Params->operands()) {
    SPIRVEntry *P = transDbgEntry(cast<DINode>(Op.get()));
    Ops.push_back(P->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::TypeTemplateParameterPack, getVoidTy(),
                          Ops);
}

// Global objects

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgGlobalVariable(const DIGlobalVariable *GV) {
  using namespace SPIRVDebug::Operand::GlobalVariable;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[NameIdx] = BM->getString(GV->getName())->getId();
  Ops[TypeIdx] = transDbgEntryRef(GV->getType())->getId();
  Ops[SourceIdx] = getSource(GV)->getId();
  Ops[LineIdx] = GV->getLine();
  Ops[ColumnIdx] = 0; // This version of DIGlobalVariable has no column number

  // Parent scope
  DIScope *Context = GV->getScope();
  SPIRVEntry *Parent = SPIRVCU;
  // Global variable may be declared in scope of a namespace or it may be a
  // static variable declared in scope of a function
  if (Context && (isa<DINamespace>(Context) || isa<DISubprogram>(Context)))
    Parent = transDbgEntry(Context);
  Ops[ParentIdx] = Parent->getId();

  Ops[LinkageNameIdx] = BM->getString(GV->getLinkageName())->getId();
  Ops[VariableIdx] = getGlobalVariable(GV)->getId();
  Ops[FlagsIdx] = transDebugFlags(GV);

  // Check if GV is the definition of previously declared static member
  if (DIDerivedType *StaticMember = GV->getStaticDataMemberDeclaration())
    Ops.push_back(transDbgEntry(StaticMember)->getId());

  return BM->addDebugInfo(SPIRVDebug::GlobalVariable, getVoidTy(), Ops);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgFunction(const DISubprogram *Func) {
  auto It = MDMap.find(Func);
  if (It != MDMap.end())
    return static_cast<SPIRVValue *>(It->second);

  // As long as indexes of FunctionDeclaration operands match with Function
  using namespace SPIRVDebug::Operand::FunctionDeclaration;
  SPIRVWordVec Ops(OperandCount);
  Ops[NameIdx] = BM->getString(Func->getName())->getId();
  Ops[TypeIdx] = transDbgEntry(Func->getType())->getId();
  Ops[SourceIdx] = getSource(Func)->getId();
  Ops[LineIdx] = Func->getLine();
  Ops[ColumnIdx] = 0; // This version of DISubprogram has no column number
  auto Scope = Func->getScope();
  if (Scope && isa<DIFile>(Scope))
    Ops[ParentIdx] = SPIRVCU->getId();
  else
    Ops[ParentIdx] = getScope(Scope)->getId();
  Ops[LinkageNameIdx] = BM->getString(Func->getLinkageName())->getId();
  Ops[FlagsIdx] = transDebugFlags(Func);

  SPIRVEntry *DebugFunc = nullptr;
  if (!Func->isDefinition()) {
    DebugFunc = BM->addDebugInfo(SPIRVDebug::FunctionDecl, getVoidTy(), Ops);
  } else {
    // Here we add operands specific function definition
    using namespace SPIRVDebug::Operand::Function;
    Ops.resize(MinOperandCount);
    Ops[ScopeLineIdx] = Func->getScopeLine();

    Ops[FunctionIdIdx] = getDebugInfoNoneId();
    for (const llvm::Function &F : M->functions()) {
      if (Func->describes(&F)) {
        SPIRVValue *SPIRVFunc = SPIRVWriter->getTranslatedValue(&F);
        assert(SPIRVFunc && "All function must be already translated");
        Ops[FunctionIdIdx] = SPIRVFunc->getId();
        break;
      }
    }

    if (DISubprogram *FuncDecl = Func->getDeclaration())
      Ops.push_back(transDbgEntry(FuncDecl)->getId());
    else
      Ops.push_back(getDebugInfoNoneId());

    DebugFunc = BM->addDebugInfo(SPIRVDebug::Function, getVoidTy(), Ops);
    MDMap.insert(std::make_pair(Func, DebugFunc));
    // Functions local variable might be not refered to anywhere else, except
    // here.
    // Just translate them.
    for (const DINode *Var : Func->getRetainedNodes())
      transDbgEntry(Var);
  }
  // If the function has template parameters the function *is* a template.
  if (DITemplateParameterArray TPA = Func->getTemplateParams()) {
    DebugFunc = transDbgTemplateParams(TPA, DebugFunc);
  }
  return DebugFunc;
}

// Location information

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgScope(const DIScope *S) {
  if (const DILexicalBlockFile *LBF = dyn_cast<DILexicalBlockFile>(S)) {
    using namespace SPIRVDebug::Operand::LexicalBlockDiscriminator;
    SPIRVWordVec Ops(OperandCount);
    Ops[SourceIdx] = getSource(S)->getId();
    Ops[DiscriminatorIdx] = LBF->getDiscriminator();
    Ops[ParentIdx] = getScope(S->getScope())->getId();
    return BM->addDebugInfo(SPIRVDebug::LexicalBlockDiscriminator, getVoidTy(),
                            Ops);
  }
  using namespace SPIRVDebug::Operand::LexicalBlock;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[SourceIdx] = getSource(S)->getId();
  Ops[ParentIdx] = getScope(S->getScope())->getId();
  if (const DILexicalBlock *LB = dyn_cast<DILexicalBlock>(S)) {
    Ops[LineIdx] = LB->getLine();
    Ops[ColumnIdx] = LB->getColumn();
  } else if (const DINamespace *NS = dyn_cast<DINamespace>(S)) {
    Ops[LineIdx] = 0;   // This version of DINamespace has no line number
    Ops[ColumnIdx] = 0; // This version of DINamespace has no column number
    Ops.push_back(BM->getString(NS->getName())->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::LexicalBlock, getVoidTy(), Ops);
}

// Generating DebugScope and DebugNoScope instructions. They can interleave with
// core instructions.
SPIRVEntry *LLVMToSPIRVDbgTran::transDebugLoc(const DebugLoc &Loc,
                                              SPIRVBasicBlock *BB,
                                              SPIRVInstruction *InsertBefore) {
  SPIRVId ExtSetId = BM->getExtInstSetId(SPIRVEIS_Debug);
  if (!Loc.get())
    return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::NoScope,
                          std::vector<SPIRVWord>(), BB, InsertBefore);

  using namespace SPIRVDebug::Operand::Scope;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[ScopeIdx] = getScope(static_cast<DIScope *>(Loc.getScope()))->getId();
  if (DILocation *IA = Loc.getInlinedAt())
    Ops.push_back(transDbgEntry(IA)->getId());
  return BM->addExtInst(getVoidTy(), ExtSetId, SPIRVDebug::Scope, Ops, BB,
                        InsertBefore);
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgInlinedAt(const DILocation *Loc) {
  using namespace SPIRVDebug::Operand::InlinedAt;
  SPIRVWordVec Ops(MinOperandCount);
  Ops[LineIdx] = Loc->getLine();
  Ops[ScopeIdx] = getScope(Loc->getScope())->getId();
  if (DILocation *IA = Loc->getInlinedAt())
    Ops.push_back(transDbgEntry(IA)->getId());
  return BM->addDebugInfo(SPIRVDebug::InlinedAt, getVoidTy(), Ops);
}

template <class T>
SPIRVExtInst *LLVMToSPIRVDbgTran::getSource(const T *DIEntry) {
  const std::string FileName = getFullPath(DIEntry);
  auto It = FileMap.find(FileName);
  if (It != FileMap.end())
    return It->second;

  using namespace SPIRVDebug::Operand::Source;
  SPIRVWordVec Ops(OperandCount);
  Ops[FileIdx] = BM->getString(FileName)->getId();
  Ops[TextIdx] = getDebugInfoNone()->getId();
  SPIRVExtInst *Source = static_cast<SPIRVExtInst *>(
      BM->addDebugInfo(SPIRVDebug::Source, getVoidTy(), Ops));
  FileMap[FileName] = Source;
  return Source;
}

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgFileType(const DIFile *F) {
  return BM->getString(getFullPath(F));
}

// Local variables

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgLocalVariable(const DILocalVariable *Var) {
  using namespace SPIRVDebug::Operand::LocalVariable;
  SPIRVWordVec Ops(MinOperandCount);

  Ops[NameIdx] = BM->getString(Var->getName())->getId();
  Ops[TypeIdx] = transDbgEntryRef(Var->getType())->getId();
  Ops[SourceIdx] = getSource(Var->getFile())->getId();
  Ops[LineIdx] = Var->getLine();
  Ops[ColumnIdx] = 0; // This version of DILocalVariable has no column number
  Ops[ParentIdx] = getScope(Var->getScope())->getId();
  Ops[FlagsIdx] = transDebugFlags(Var);
  if (SPIRVWord ArgNumber = Var->getArg())
    Ops.push_back(ArgNumber);
  return BM->addDebugInfo(SPIRVDebug::LocalVariable, getVoidTy(), Ops);
}

// DWARF Operations and expressions

SPIRVEntry *LLVMToSPIRVDbgTran::transDbgExpression(const DIExpression *Expr) {
  SPIRVWordVec Operations;
  for (unsigned I = 0, N = Expr->getNumElements(); I < N; ++I) {
    using namespace SPIRVDebug::Operand::Operation;
    auto DWARFOpCode = static_cast<dwarf::LocationAtom>(Expr->getElement(I));
    SPIRVDebug::ExpressionOpCode OC =
        SPIRV::DbgExpressionOpCodeMap::map(DWARFOpCode);
    assert(OpCountMap.find(OC) != OpCountMap.end() &&
           "unhandled opcode found in DIExpression");
    unsigned OpCount = OpCountMap[OC];
    SPIRVWordVec Op(OpCount);
    Op[OpCodeIdx] = OC;
    for (unsigned J = 1; J < OpCount; ++J)
      Op[J] = Expr->getElement(++I);
    auto *Operation = BM->addDebugInfo(SPIRVDebug::Operation, getVoidTy(), Op);
    Operations.push_back(Operation->getId());
  }
  return BM->addDebugInfo(SPIRVDebug::Expression, getVoidTy(), Operations);
}

// Imported entries (C++ using directive)

SPIRVEntry *
LLVMToSPIRVDbgTran::transDbgImportedEntry(const DIImportedEntity *IE) {
  using namespace SPIRVDebug::Operand::ImportedEntity;
  SPIRVWordVec Ops(OperandCount);
  auto Tag = static_cast<dwarf::Tag>(IE->getTag());
  Ops[NameIdx] = BM->getString(IE->getName())->getId();
  Ops[TagIdx] = SPIRV::DbgImportedEntityMap::map(Tag);
  Ops[SourceIdx] = getSource(IE->getFile())->getId();
  Ops[EntityIdx] = transDbgEntryRef(IE->getEntity())->getId();
  Ops[LineIdx] = IE->getLine();
  Ops[ColumnIdx] = 0; // This version of DIImportedEntity has no column number
  Ops[ParentIdx] = getScope(IE->getScope())->getId();
  return BM->addDebugInfo(SPIRVDebug::ImportedEntity, getVoidTy(), Ops);
}
