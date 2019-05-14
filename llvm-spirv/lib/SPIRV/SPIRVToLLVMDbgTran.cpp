//===- SPIRVToLLVMDbgTran.cpp - Converts debug info to LLVM -----*- C++ -*-===//
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
// This file implements translation of debug info from SPIR-V to LLVM metadata
//
//===----------------------------------------------------------------------===//

#include "SPIRVToLLVMDbgTran.h"
#include "SPIRVEntry.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVReader.h"
#include "SPIRVType.h"

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Module.h"

using namespace std;
using namespace SPIRVDebug::Operand;

namespace SPIRV {

SPIRVToLLVMDbgTran::SPIRVToLLVMDbgTran(SPIRVModule *TBM, Module *TM,
                                       SPIRVToLLVM *Reader)
    : BM(TBM), M(TM), Builder(*M), SPIRVReader(Reader) {
  Enable = BM->hasDebugInfo();
}

void SPIRVToLLVMDbgTran::addDbgInfoVersion() {
  if (!Enable)
    return;
  M->addModuleFlag(Module::Warning, "Debug Info Version",
                   DEBUG_METADATA_VERSION);
}

DIFile *SPIRVToLLVMDbgTran::getDIFile(const string &FileName) {
  return getOrInsert(FileMap, FileName, [=]() {
    SplitFileName Split(FileName);
    if (!Split.BaseName.empty())
      return Builder.createFile(Split.BaseName, Split.Path);
    return static_cast<DIFile *>(nullptr);
  });
}

SPIRVExtInst *SPIRVToLLVMDbgTran::getDbgInst(const SPIRVId Id) {
  SPIRVEntry *E = BM->getEntry(Id);
  if (isa<OpExtInst>(E)) {
    SPIRVExtInst *EI = static_cast<SPIRVExtInst *>(E);
    if (EI->getExtSetKind() == SPIRV::SPIRVEIS_Debug)
      return EI;
  }
  return nullptr;
}

StringRef SPIRVToLLVMDbgTran::getString(const SPIRVId Id) {
  SPIRVString *String = BM->get<SPIRVString>(Id);
  assert(String && "Invalid string");
  return String->getStr();
}

void SPIRVToLLVMDbgTran::transDbgInfo(const SPIRVValue *SV, Value *V) {
  // A constant sampler does not have a corresponding SPIRVInstruction.
  if (SV->getOpCode() == OpConstantSampler)
    return;

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    const SPIRVInstruction *SI = static_cast<const SPIRVInstruction *>(SV);
    I->setDebugLoc(transDebugScope(SI));
  }
}

DIScope *SPIRVToLLVMDbgTran::getScope(const SPIRVEntry *ScopeInst) {
  if (ScopeInst->getOpCode() == OpString)
    return getDIFile(static_cast<const SPIRVString *>(ScopeInst)->getStr());
  return transDebugInst<DIScope>(static_cast<const SPIRVExtInst *>(ScopeInst));
}

DICompileUnit *
SPIRVToLLVMDbgTran::transCompileUnit(const SPIRVExtInst *DebugInst) {
  const SPIRVWordVec &Ops = DebugInst->getArguments();

  using namespace SPIRVDebug::Operand::CompilationUnit;
  assert(Ops.size() == OperandCount && "Invalid number of operands");
  M->addModuleFlag(llvm::Module::Warning, "Dwarf Version",
                   Ops[DWARFVersionIdx]);
  SPIRVExtInst *Source = BM->get<SPIRVExtInst>(Ops[SourceIdx]);
  SPIRVId FileId = Source->getArguments()[SPIRVDebug::Operand::Source::FileIdx];
  std::string File = getString(FileId);
  unsigned SourceLang = Ops[LanguageIdx];
  CU = Builder.createCompileUnit(SourceLang, getDIFile(File), "spirv", false,
                                 "", 0);
  return CU;
}

DIBasicType *SPIRVToLLVMDbgTran::transTypeBasic(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeBasic;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() == OperandCount && "Invalid number of operands");
  StringRef Name = getString(Ops[NameIdx]);
  auto Tag = static_cast<SPIRVDebug::EncodingTag>(Ops[EncodingIdx]);
  unsigned Encoding = SPIRV::DbgEncodingMap::rmap(Tag);
  if (Encoding == 0)
    return Builder.createUnspecifiedType(Name);
  uint64_t Size = BM->get<SPIRVConstant>(Ops[SizeIdx])->getZExtIntValue();
  return Builder.createBasicType(Name, Size, Encoding);
}

DIDerivedType *
SPIRVToLLVMDbgTran::transTypeQualifier(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeQualifier;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() == OperandCount && "Invalid number of operands");
  DIType *BaseTy =
      transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[BaseTypeIdx]));
  unsigned Tag = SPIRV::DbgTypeQulifierMap::rmap(
      static_cast<SPIRVDebug::TypeQualifierTag>(Ops[QualifierIdx]));
  return Builder.createQualifiedType(Tag, BaseTy);
}

DIType *SPIRVToLLVMDbgTran::transTypePointer(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypePointer;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() == OperandCount && "Invalid number of operands");
  DIType *PointeeTy = nullptr;
  if (BM->getEntry(Ops[BaseTypeIdx])->getOpCode() != OpTypeVoid)
    PointeeTy = transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[BaseTypeIdx]));
  Optional<unsigned> AS;
  if (Ops[StorageClassIdx] != ~0U) {
    auto SC = static_cast<SPIRVStorageClassKind>(Ops[StorageClassIdx]);
    AS = SPIRSPIRVAddrSpaceMap::rmap(SC);
  }
  DIType *Ty;
  SPIRVWord Flags = Ops[FlagsIdx];
  if (Flags & SPIRVDebug::FlagIsLValueReference)
    Ty = Builder.createReferenceType(dwarf::DW_TAG_reference_type, PointeeTy, 0,
                                     0, AS);
  else if (Flags & SPIRVDebug::FlagIsRValueReference)
    Ty = Builder.createReferenceType(dwarf::DW_TAG_rvalue_reference_type,
                                     PointeeTy, 0, 0, AS);
  else
    Ty = Builder.createPointerType(PointeeTy, BM->getAddressingModel() * 32, 0,
                                   AS);

  if (Flags & SPIRVDebug::FlagIsObjectPointer)
    Ty = Builder.createObjectPointerType(Ty);
  else if (Flags & SPIRVDebug::FlagIsArtificial)
    Ty = Builder.createArtificialType(Ty);

  return Ty;
}

DICompositeType *
SPIRVToLLVMDbgTran::transTypeArray(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeArray;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");
  DIType *BaseTy =
      transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[BaseTypeIdx]));
  size_t TotalCount = 1;
  SmallVector<llvm::Metadata *, 8> Subscripts;
  for (size_t I = ComponentCountIdx, E = Ops.size(); I < E; ++I) {
    SPIRVConstant *C = BM->get<SPIRVConstant>(Ops[I]);
    int64_t Count = static_cast<int64_t>(C->getZExtIntValue());
    Subscripts.push_back(Builder.getOrCreateSubrange(0, Count));
    TotalCount *= static_cast<uint64_t>(Count);
  }
  DINodeArray SubscriptArray = Builder.getOrCreateArray(Subscripts);
  size_t Size = BaseTy->getSizeInBits() * TotalCount;
  return Builder.createArrayType(Size, 0 /*align*/, BaseTy, SubscriptArray);
}

DICompositeType *
SPIRVToLLVMDbgTran::transTypeVector(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeVector;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");
  DIType *BaseTy =
      transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[BaseTypeIdx]));
  SPIRVWord Count = Ops[ComponentCountIdx];
  uint64_t Size = BaseTy->getSizeInBits() * Count;

  SmallVector<llvm::Metadata *, 8> Subscripts;
  Subscripts.push_back(Builder.getOrCreateSubrange(0, Count));
  DINodeArray SubscriptArray = Builder.getOrCreateArray(Subscripts);

  return Builder.createVectorType(Size, 0 /*align*/, BaseTy, SubscriptArray);
}

DICompositeType *
SPIRVToLLVMDbgTran::transTypeComposite(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeComposite;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  StringRef Name = getString(Ops[NameIdx]);
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DIScope *ParentScope = getScope(BM->getEntry(Ops[ParentIdx]));
  uint64_t Size = BM->get<SPIRVConstant>(Ops[SizeIdx])->getZExtIntValue();

  uint64_t Align = 0;
  DIType *DerivedFrom = nullptr;
  StringRef Identifier;
  SPIRVEntry *UniqId = BM->get<SPIRVEntry>(Ops[LinkageNameIdx]);
  if (UniqId->getOpCode() == OpString)
    Identifier = static_cast<SPIRVString *>(UniqId)->getStr();

  DINode::DIFlags Flags = DINode::FlagZero;
  if (Ops[FlagsIdx] & SPIRVDebug::FlagIsFwdDecl)
    Flags |= DINode::FlagFwdDecl;

  DICompositeType *CT = nullptr;
  switch (Ops[TagIdx]) {
  case SPIRVDebug::Class:
    CT = Builder.createClassType(
        ParentScope, Name, File, LineNo, Size, Align, 0, Flags, DerivedFrom,
        DINodeArray() /*elements*/, nullptr /*VTableHolder*/,
        nullptr /*TemplateParams*/, Identifier);
    break;
  case SPIRVDebug::Structure:
    CT = Builder.createStructType(ParentScope, Name, File, LineNo, Size, Align,
                                  Flags, DerivedFrom,
                                  DINodeArray() /*elements*/, 0 /*RunTimeLang*/,
                                  nullptr /*VTableHolder*/, Identifier);
    break;
  case SPIRVDebug::Union:
    CT = Builder.createUnionType(ParentScope, Name, File, LineNo, Size, Align,
                                 Flags, DINodeArray(), 0 /*RuntimrLang*/,
                                 Identifier);
    break;
  default:
    llvm_unreachable("Unexpected composite type");
    break;
  }
  DebugInstCache[DebugInst] = CT;
  SmallVector<llvm::Metadata *, 8> EltTys;
  for (size_t I = FirstMemberIdx; I < Ops.size(); ++I) {
    EltTys.push_back(transDebugInst(BM->get<SPIRVExtInst>(Ops[I])));
  }
  DINodeArray Elements = Builder.getOrCreateArray(EltTys);
  Builder.replaceArrays(CT, Elements);
  assert(CT && "Composite type translation failed.");
  return CT;
}

DINode *SPIRVToLLVMDbgTran::transTypeMember(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeMember;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  StringRef Name = getString(Ops[NameIdx]);
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  DIType *BaseType =
      transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[TypeIdx]));
  uint64_t OffsetInBits =
      BM->get<SPIRVConstant>(Ops[OffsetIdx])->getZExtIntValue();
  unsigned SPIRVFlags = Ops[FlagsIdx];
  DINode::DIFlags Flags = DINode::FlagZero;
  if ((SPIRVDebug::FlagAccess & SPIRVFlags) == SPIRVDebug::FlagIsPublic) {
    Flags |= DINode::FlagPublic;
  } else if (SPIRVFlags & SPIRVDebug::FlagIsProtected) {
    Flags |= DINode::FlagProtected;
  } else if (SPIRVFlags & SPIRVDebug::FlagIsPrivate) {
    Flags |= DINode::FlagPrivate;
  }
  if (SPIRVFlags & SPIRVDebug::FlagIsStaticMember)
    Flags |= DINode::FlagStaticMember;

  if (Flags & DINode::FlagStaticMember && Ops.size() > MinOperandCount) {
    SPIRVValue *ConstVal = BM->get<SPIRVValue>(Ops[ValueIdx]);
    assert(isConstantOpCode(ConstVal->getOpCode()) &&
           "Static member must be a constant");
    llvm::Value *Val = SPIRVReader->transValue(ConstVal, nullptr, nullptr);
    return Builder.createStaticMemberType(Scope, Name, File, LineNo, BaseType,
                                          Flags, cast<llvm::Constant>(Val));
  }
  uint64_t Size = BM->get<SPIRVConstant>(Ops[SizeIdx])->getZExtIntValue();
  uint64_t Alignment = 0;

  return Builder.createMemberType(Scope, Name, File, LineNo, Size, Alignment,
                                  OffsetInBits, Flags, BaseType);
}

DINode *SPIRVToLLVMDbgTran::transTypeEnum(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeEnum;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  StringRef Name = getString(Ops[NameIdx]);
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  uint64_t SizeInBits = BM->get<SPIRVConstant>(Ops[SizeIdx])->getZExtIntValue();
  unsigned AlignInBits = 0;
  SPIRVWord Flags = Ops[FlagsIdx];
  if (Flags & SPIRVDebug::FlagIsFwdDecl) {
    return Builder.createForwardDecl(dwarf::DW_TAG_enumeration_type, Name,
                                     Scope, File, LineNo, AlignInBits,
                                     SizeInBits);
  } else {
    SmallVector<llvm::Metadata *, 16> Elts;
    for (size_t I = FirstEnumeratorIdx, E = Ops.size(); I < E; I += 2) {
      uint64_t Val = BM->get<SPIRVConstant>(Ops[I])->getZExtIntValue();
      StringRef Name = getString(Ops[I + 1]);
      Elts.push_back(Builder.createEnumerator(Name, Val));
    }
    DINodeArray Enumerators = Builder.getOrCreateArray(Elts);
    DIType *UnderlyingType = nullptr;
    SPIRVEntry *E = BM->getEntry(Ops[UnderlyingTypeIdx]);
    if (!isa<OpTypeVoid>(E))
      UnderlyingType = transDebugInst<DIType>(static_cast<SPIRVExtInst *>(E));
    return Builder.createEnumerationType(Scope, Name, File, LineNo, SizeInBits,
                                         AlignInBits, Enumerators,
                                         UnderlyingType, "", UnderlyingType);
  }
}

DINode *SPIRVToLLVMDbgTran::transTypeFunction(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeFunction;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  SPIRVWord SPIRVFlags = Ops[FlagsIdx];
  DINode::DIFlags Flags = DINode::FlagZero;
  if (SPIRVFlags & SPIRVDebug::FlagIsLValueReference)
    Flags |= llvm::DINode::FlagLValueReference;
  if (SPIRVFlags & SPIRVDebug::FlagIsRValueReference)
    Flags |= llvm::DINode::FlagRValueReference;

  SPIRVEntry *E = BM->getEntry(Ops[ReturnTypeIdx]);
  MDNode *RT = isa<OpTypeVoid>(E)
                   ? nullptr
                   : transDebugInst(BM->get<SPIRVExtInst>(Ops[ReturnTypeIdx]));
  SmallVector<llvm::Metadata *, 16> Elements{RT};
  for (size_t I = FirstParameterIdx, E = Ops.size(); I < E; ++I) {
    SPIRVEntry *P = BM->getEntry(Ops[I]);
    MDNode *Param = isa<OpTypeVoid>(P)
                        ? nullptr
                        : transDebugInst(BM->get<SPIRVExtInst>(Ops[I]));

    Elements.push_back(Param);
  }
  DITypeRefArray ArgTypes = Builder.getOrCreateTypeArray(Elements);
  return Builder.createSubroutineType(ArgTypes, Flags);
}

DINode *
SPIRVToLLVMDbgTran::transTypePtrToMember(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::PtrToMember;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");
  SPIRVExtInst *Member = BM->get<SPIRVExtInst>(Ops[MemberTypeIdx]);
  DIType *PointeeTy = transDebugInst<DIType>(Member);
  SPIRVExtInst *ContainingTy = BM->get<SPIRVExtInst>(Ops[ParentIdx]);
  DIType *BaseTy = transDebugInst<DIType>(ContainingTy);
  return Builder.createMemberPointerType(PointeeTy, BaseTy, 0);
}

DINode *SPIRVToLLVMDbgTran::transLexicalBlock(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::LexicalBlock;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  DIScope *ParentScope = getScope(BM->getEntry(Ops[ParentIdx]));
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  if (Ops.size() > NameIdx) {
    StringRef Name = getString(Ops[NameIdx]);
    return Builder.createNameSpace(ParentScope, Name,
                                   false /*inlined namespace*/);
  }
  unsigned Column = Ops[ColumnIdx];
  return Builder.createLexicalBlock(ParentScope, File, LineNo, Column);
}

DINode *SPIRVToLLVMDbgTran::transLexicalBlockDiscriminator(
    const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::LexicalBlockDiscriminator;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned Disc = Ops[DiscriminatorIdx];
  DIScope *ParentScope = getScope(BM->getEntry(Ops[ParentIdx]));
  return Builder.createLexicalBlockFile(ParentScope, File, Disc);
}

DINode *SPIRVToLLVMDbgTran::transFunction(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::Function;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  StringRef Name = getString(Ops[NameIdx]);
  DISubroutineType *Ty =
      transDebugInst<DISubroutineType>(BM->get<SPIRVExtInst>(Ops[TypeIdx]));
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  StringRef LinkageName = getString(Ops[LinkageNameIdx]);

  SPIRVWord SPIRVDebugFlags = Ops[FlagsIdx];
  DINode::DIFlags Flags = DINode::FlagZero;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsArtificial)
    Flags |= llvm::DINode::FlagArtificial;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsExplicit)
    Flags |= llvm::DINode::FlagExplicit;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsPrototyped)
    Flags |= llvm::DINode::FlagPrototyped;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsLValueReference)
    Flags |= llvm::DINode::FlagLValueReference;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsRValueReference)
    Flags |= llvm::DINode::FlagRValueReference;
  if ((SPIRVDebugFlags & SPIRVDebug::FlagAccess) == SPIRVDebug::FlagIsPublic)
    Flags |= llvm::DINode::FlagPublic;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsProtected)
    Flags |= llvm::DINode::FlagProtected;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsPrivate)
    Flags |= llvm::DINode::FlagPrivate;

  bool IsDefinition = SPIRVDebugFlags & SPIRVDebug::FlagIsDefinition;
  bool IsOptimized = SPIRVDebugFlags & SPIRVDebug::FlagIsOptimized;
  bool IsLocal = SPIRVDebugFlags & SPIRVDebug::FlagIsLocal;
  bool IsMainSubprogram =
      BM->isEntryPoint(spv::ExecutionModelKernel, Ops[FunctionIdIdx]);
  DISubprogram::DISPFlags SPFlags =
      DISubprogram::toSPFlags(IsLocal, IsDefinition, IsOptimized,
                              DISubprogram::SPFlagNonvirtual, IsMainSubprogram);

  unsigned ScopeLine = Ops[ScopeLineIdx];

  // Function declaration descriptor
  DISubprogram *FD = nullptr;
  if (Ops.size() > DeclarationIdx) {
    FD = transDebugInst<DISubprogram>(
        BM->get<SPIRVExtInst>(Ops[DeclarationIdx]));
  }

  // Here we create fake array of template parameters. If it was plain nullptr,
  // the template parameter operand would be removed in DISubprogram::getImpl.
  // But we want it to be there, because if there is DebugTemplate instruction
  // refering to this function, TransTemplate method must be able to replace the
  // template parameter operand, thus it must be in the operands list.
  SmallVector<llvm::Metadata *, 8> Elts;
  DINodeArray TParams = Builder.getOrCreateArray(Elts);
  llvm::DITemplateParameterArray TParamsArray = TParams.get();

  DISubprogram *DIS = nullptr;
  if ((isa<DICompositeType>(Scope) || isa<DINamespace>(Scope)) && !IsDefinition)
    DIS = Builder.createMethod(Scope, Name, LinkageName, File, LineNo, Ty, 0, 0,
                               nullptr, Flags, SPFlags, TParamsArray);
  else
    DIS = Builder.createFunction(Scope, Name, LinkageName, File, LineNo, Ty,
                                 ScopeLine, Flags, SPFlags, TParamsArray, FD);
  DebugInstCache[DebugInst] = DIS;
  SPIRVId RealFuncId = Ops[FunctionIdIdx];
  FuncMap[RealFuncId] = DIS;

  // Function.
  SPIRVEntry *E = BM->getEntry(Ops[FunctionIdIdx]);
  if (E->getOpCode() == OpFunction) {
    SPIRVFunction *BF = static_cast<SPIRVFunction *>(E);
    llvm::Function *F = SPIRVReader->transFunction(BF);
    assert(F && "Translation of function failed!");
    if (!F->hasMetadata())
      F->setMetadata("dbg", DIS);
  }
  return DIS;
}

DINode *SPIRVToLLVMDbgTran::transFunctionDecl(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::FunctionDeclaration;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() == OperandCount && "Invalid number of operands");
  // Scope
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  StringRef Name = getString(Ops[NameIdx]);
  StringRef LinkageName = getString(Ops[LinkageNameIdx]);
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DISubroutineType *Ty =
      transDebugInst<DISubroutineType>(BM->get<SPIRVExtInst>(Ops[TypeIdx]));

  SPIRVWord SPIRVDebugFlags = Ops[FlagsIdx];
  bool IsDefinition = SPIRVDebugFlags & SPIRVDebug::FlagIsDefinition;
  bool IsOptimized = SPIRVDebugFlags & SPIRVDebug::FlagIsOptimized;
  bool IsLocal = SPIRVDebugFlags & SPIRVDebug::FlagIsLocal;
  DINode::DIFlags Flags = DINode::FlagZero;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsArtificial)
    Flags |= llvm::DINode::FlagArtificial;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsExplicit)
    Flags |= llvm::DINode::FlagExplicit;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsPrototyped)
    Flags |= llvm::DINode::FlagPrototyped;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsLValueReference)
    Flags |= llvm::DINode::FlagLValueReference;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsRValueReference)
    Flags |= llvm::DINode::FlagRValueReference;
  if ((SPIRVDebugFlags & SPIRVDebug::FlagAccess) == SPIRVDebug::FlagIsPublic)
    Flags |= llvm::DINode::FlagPublic;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsProtected)
    Flags |= llvm::DINode::FlagProtected;
  if (SPIRVDebugFlags & SPIRVDebug::FlagIsPrivate)
    Flags |= llvm::DINode::FlagPrivate;

  // Here we create fake array of template parameters. If it was plain nullptr,
  // the template parameter operand would be removed in DISubprogram::getImpl.
  // But we want it to be there, because if there is DebugTemplate instruction
  // refering to this function, TransTemplate method must be able to replace the
  // template parameter operand, thus it must be in the operands list.
  SmallVector<llvm::Metadata *, 8> Elts;
  DINodeArray TParams = Builder.getOrCreateArray(Elts);
  llvm::DITemplateParameterArray TParamsArray = TParams.get();

  DISubprogram *DIS = nullptr;
  DISubprogram::DISPFlags SPFlags =
      DISubprogram::toSPFlags(IsLocal, IsDefinition, IsOptimized);
  if (isa<DICompositeType>(Scope) || isa<DINamespace>(Scope))
    DIS = Builder.createMethod(Scope, Name, LinkageName, File, LineNo, Ty, 0, 0,
                               nullptr, Flags, SPFlags, TParamsArray);
  else {
    // Since a function declaration doesn't have any retained nodes, resolve
    // the temporary placeholder for them immediately.
    DIS = Builder.createTempFunctionFwdDecl(Scope, Name, LinkageName, File,
                                            LineNo, Ty, 0, Flags, SPFlags,
                                            TParamsArray);
    llvm::TempMDNode FwdDecl(cast<llvm::MDNode>(DIS));
    DIS = Builder.replaceTemporary(std::move(FwdDecl), DIS);
  }
  DebugInstCache[DebugInst] = DIS;

  return DIS;
}

MDNode *SPIRVToLLVMDbgTran::transGlobalVariable(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::GlobalVariable;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  StringRef Name = getString(Ops[NameIdx]);
  DIType *Ty = transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[TypeIdx]));
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DIScope *Parent = getScope(BM->getEntry(Ops[ParentIdx]));
  StringRef LinkageName = getString(Ops[LinkageNameIdx]);

  DIDerivedType *StaticMemberDecl = nullptr;
  if (Ops.size() > MinOperandCount) {
    StaticMemberDecl = transDebugInst<DIDerivedType>(
        BM->get<SPIRVExtInst>(Ops[StaticMemberDeclarationIdx]));
  }
  bool IsLocal = Ops[FlagsIdx] & SPIRVDebug::FlagIsLocal;
  bool IsDefinition = Ops[FlagsIdx] & SPIRVDebug::FlagIsDefinition;
  MDNode *VarDecl = nullptr;
  if (IsDefinition) {
    VarDecl = Builder.createGlobalVariableExpression(Parent, Name, LinkageName,
                                                     File, LineNo, Ty, IsLocal,
                                                     nullptr, StaticMemberDecl);
  } else {
    VarDecl = Builder.createTempGlobalVariableFwdDecl(
        Parent, Name, LinkageName, File, LineNo, Ty, IsLocal, StaticMemberDecl);
    // replaceAllUsesWith call makes VarDecl non-temp.
    // Otherwise DIBuilder will crash at finalization.
    llvm::TempMDNode TMP(VarDecl);
    Builder.replaceTemporary(std::move(TMP), VarDecl);
  }
  // If the variable has no initializer Ops[VariableIdx] is OpDebugInfoNone.
  // Otherwise Ops[VariableIdx] may be a global variable or a constant(C++
  // static const).
  if (VarDecl && !getDbgInst<SPIRVDebug::DebugInfoNone>(Ops[VariableIdx])) {
    SPIRVValue *V = BM->get<SPIRVValue>(Ops[VariableIdx]);
    Value *Var = SPIRVReader->transValue(V, nullptr, nullptr);
    llvm::GlobalVariable *GV = dyn_cast_or_null<llvm::GlobalVariable>(Var);
    if (GV && !GV->hasMetadata())
      GV->addMetadata("dbg", *VarDecl);
  }
  return VarDecl;
}

DINode *SPIRVToLLVMDbgTran::transLocalVariable(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::LocalVariable;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");

  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  StringRef Name = getString(Ops[NameIdx]);
  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  DIType *Ty = transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[TypeIdx]));
  DINode::DIFlags Flags = DINode::FlagZero;
  if (Ops[FlagsIdx] & SPIRVDebug::FlagIsArtificial)
    Flags |= DINode::FlagArtificial;
  if (Ops[FlagsIdx] & SPIRVDebug::FlagIsObjectPointer)
    Flags |= DINode::FlagObjectPointer;

  if (Ops.size() > ArgNumberIdx)
    return Builder.createParameterVariable(Scope, Name, Ops[ArgNumberIdx], File,
                                           LineNo, Ty, true, Flags);
  return Builder.createAutoVariable(Scope, Name, File, LineNo, Ty, true, Flags);
}

DINode *SPIRVToLLVMDbgTran::transTypedef(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::Typedef;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");

  DIFile *File = getFile(Ops[SourceIdx]);
  unsigned LineNo = Ops[LineIdx];
  StringRef Alias = getString(Ops[NameIdx]);
  SPIRVEntry *TypeInst = BM->getEntry(Ops[BaseTypeIdx]);
  DIType *Ty = transDebugInst<DIType>(static_cast<SPIRVExtInst *>(TypeInst));
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  assert(Scope && "Typedef should have a parent scope");
  return Builder.createTypedef(Ty, Alias, File, LineNo, Scope);
}

DINode *SPIRVToLLVMDbgTran::transInheritance(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TypeInheritance;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");
  DIType *Parent =
      transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[ParentIdx]));
  DIType *Child = transDebugInst<DIType>(BM->get<SPIRVExtInst>(Ops[ChildIdx]));
  DINode::DIFlags Flags = DINode::FlagZero;
  if ((Ops[FlagsIdx] & SPIRVDebug::FlagAccess) == SPIRVDebug::FlagIsPublic)
    Flags |= llvm::DINode::FlagPublic;
  if ((Ops[FlagsIdx] & SPIRVDebug::FlagAccess) == SPIRVDebug::FlagIsProtected)
    Flags |= llvm::DINode::FlagProtected;
  if ((Ops[FlagsIdx] & SPIRVDebug::FlagAccess) == SPIRVDebug::FlagIsPrivate)
    Flags |= llvm::DINode::FlagPrivate;
  uint64_t Offset = BM->get<SPIRVConstant>(Ops[OffsetIdx])->getZExtIntValue();
  return Builder.createInheritance(Child, Parent, Offset, 0, Flags);
}

DINode *
SPIRVToLLVMDbgTran::transTemplateParameter(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TemplateParameter;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");
  StringRef Name = getString(Ops[NameIdx]);
  SPIRVEntry *ActualType = BM->getEntry(Ops[TypeIdx]);
  DIType *Ty = nullptr;
  if (!isa<OpTypeVoid>(ActualType))
    Ty = transDebugInst<DIType>(static_cast<SPIRVExtInst *>(ActualType));
  DIScope *Context = nullptr;
  if (!getDbgInst<SPIRVDebug::DebugInfoNone>(Ops[ValueIdx])) {
    SPIRVValue *Val = BM->get<SPIRVValue>(Ops[ValueIdx]);
    Value *V = SPIRVReader->transValue(Val, nullptr, nullptr);
    return Builder.createTemplateValueParameter(Context, Name, Ty,
                                                cast<Constant>(V));
  }
  return Builder.createTemplateTypeParameter(Context, Name, Ty);
}

DINode *SPIRVToLLVMDbgTran::transTemplateTemplateParameter(
    const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TemplateTemplateParameter;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");
  StringRef Name = getString(Ops[NameIdx]);
  StringRef TemplName = getString(Ops[TemplateNameIdx]);
  DIScope *Context = nullptr;
  return Builder.createTemplateTemplateParameter(Context, Name, nullptr,
                                                 TemplName);
}

DINode *
SPIRVToLLVMDbgTran::transTemplateParameterPack(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::TemplateParameterPack;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");
  StringRef Name = getString(Ops[NameIdx]);
  SmallVector<llvm::Metadata *, 8> Elts;
  for (size_t I = FirstParameterIdx, E = Ops.size(); I < E; ++I) {
    Elts.push_back(transDebugInst(BM->get<SPIRVExtInst>(Ops[I])));
  }
  DINodeArray Pack = Builder.getOrCreateArray(Elts);
  DIScope *Context = nullptr;
  return Builder.createTemplateParameterPack(Context, Name, nullptr, Pack);
}

MDNode *SPIRVToLLVMDbgTran::transTemplate(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::Template;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  const size_t NumOps = Ops.size();
  assert(NumOps >= MinOperandCount && "Invalid number of operands");

  auto *Templ = BM->get<SPIRVExtInst>(Ops[TargetIdx]);
  MDNode *D = transDebugInst(Templ);

  SmallVector<llvm::Metadata *, 8> Elts;
  for (size_t I = FirstParameterIdx; I < NumOps; ++I) {
    Elts.push_back(transDebugInst(BM->get<SPIRVExtInst>(Ops[I])));
  }
  DINodeArray TParams = Builder.getOrCreateArray(Elts);

  if (DICompositeType *Comp = dyn_cast<DICompositeType>(D)) {
    Builder.replaceArrays(Comp, Comp->getElements(), TParams);
    return Comp;
  }
  if (isa<DISubprogram>(D)) {
    // This constant matches with one used in
    // DISubprogram::getRawTemplateParams()
    const unsigned TemplateParamsIndex = 9;
    D->replaceOperandWith(TemplateParamsIndex, TParams.get());
    return D;
  }
  llvm_unreachable("Invalid template");
}

DINode *SPIRVToLLVMDbgTran::transImportedEntry(const SPIRVExtInst *DebugInst) {
  using namespace SPIRVDebug::Operand::ImportedEntity;
  const SPIRVWordVec &Ops = DebugInst->getArguments();
  assert(Ops.size() >= OperandCount && "Invalid number of operands");
  DIScope *Scope = getScope(BM->getEntry(Ops[ParentIdx]));
  unsigned Line = Ops[LineIdx];
  DIFile *File = getFile(Ops[SourceIdx]);
  auto *Entity = transDebugInst<DINode>(BM->get<SPIRVExtInst>(Ops[EntityIdx]));
  if (Ops[TagIdx] == SPIRVDebug::ImportedModule) {
    if (DIImportedEntity *IE = dyn_cast<DIImportedEntity>(Entity))
      return Builder.createImportedModule(Scope, IE, File, Line);
    if (DINamespace *NS = dyn_cast<DINamespace>(Entity))
      return Builder.createImportedModule(Scope, NS, File, Line);
  }
  if (Ops[TagIdx] == SPIRVDebug::ImportedDeclaration) {
    StringRef Name = getString(Ops[NameIdx]);
    if (DIGlobalVariableExpression *GVE =
            dyn_cast<DIGlobalVariableExpression>(Entity))
      return Builder.createImportedDeclaration(Scope, GVE->getVariable(), File,
                                               Line, Name);
    return Builder.createImportedDeclaration(Scope, Entity, File, Line, Name);
  }
  llvm_unreachable("Unexpected kind of imported entity!");
}

MDNode *SPIRVToLLVMDbgTran::transExpression(const SPIRVExtInst *DebugInst) {
  const SPIRVWordVec &Args = DebugInst->getArguments();
  std::vector<int64_t> Ops;
  for (SPIRVId A : Args) {
    SPIRVExtInst *O = BM->get<SPIRVExtInst>(A);
    const SPIRVWordVec &Operands = O->getArguments();
    auto OpCode = static_cast<SPIRVDebug::ExpressionOpCode>(Operands[0]);
    Ops.push_back(SPIRV::DbgExpressionOpCodeMap::rmap(OpCode));
    for (unsigned I = 1, E = Operands.size(); I < E; ++I) {
      Ops.push_back(Operands[I]);
    }
  }
  ArrayRef<int64_t> Addr(Ops.data(), Ops.size());
  return Builder.createExpression(Addr);
}

MDNode *SPIRVToLLVMDbgTran::transDebugInstImpl(const SPIRVExtInst *DebugInst) {
  switch (DebugInst->getExtOp()) {
  case SPIRVDebug::DebugInfoNone:
    return nullptr;

  case SPIRVDebug::CompilationUnit:
    return transCompileUnit(DebugInst);

  case SPIRVDebug::TypeBasic:
    return transTypeBasic(DebugInst);

  case SPIRVDebug::TypeQualifier:
    return transTypeQualifier(DebugInst);

  case SPIRVDebug::TypePointer:
    return transTypePointer(DebugInst);

  case SPIRVDebug::TypeArray:
    return transTypeArray(DebugInst);

  case SPIRVDebug::TypeVector:
    return transTypeVector(DebugInst);

  case SPIRVDebug::TypeComposite:
    return transTypeComposite(DebugInst);

  case SPIRVDebug::TypeMember:
    return transTypeMember(DebugInst);

  case SPIRVDebug::TypePtrToMember:
    return transTypePtrToMember(DebugInst);

  case SPIRVDebug::TypeEnum:
    return transTypeEnum(DebugInst);

  case SPIRVDebug::TypeFunction:
    return transTypeFunction(DebugInst);

  case SPIRVDebug::LexicalBlock:
    return transLexicalBlock(DebugInst);

  case SPIRVDebug::LexicalBlockDiscriminator:
    return transLexicalBlockDiscriminator(DebugInst);

  case SPIRVDebug::Function:
    return transFunction(DebugInst);

  case SPIRVDebug::FunctionDecl:
    return transFunctionDecl(DebugInst);

  case SPIRVDebug::GlobalVariable:
    return transGlobalVariable(DebugInst);

  case SPIRVDebug::LocalVariable:
    return transLocalVariable(DebugInst);

  case SPIRVDebug::Typedef:
    return transTypedef(DebugInst);

  case SPIRVDebug::InlinedAt:
    return transDebugInlined(DebugInst);

  case SPIRVDebug::Inheritance:
    return transInheritance(DebugInst);

  case SPIRVDebug::TypeTemplateParameter:
    return transTemplateParameter(DebugInst);

  case SPIRVDebug::TypeTemplateTemplateParameter:
    return transTemplateTemplateParameter(DebugInst);

  case SPIRVDebug::TypeTemplateParameterPack:
    return transTemplateParameterPack(DebugInst);

  case SPIRVDebug::TypeTemplate:
    return transTemplate(DebugInst);

  case SPIRVDebug::ImportedEntity:
    return transImportedEntry(DebugInst);

  case SPIRVDebug::Operation: // To be translated with transExpression
  case SPIRVDebug::Source:    // To be used by other instructions
    return nullptr;

  case SPIRVDebug::Expression:
    return transExpression(DebugInst);

  default:
    llvm_unreachable("Not implemented SPIR-V debug instruction!");
  }
}

Instruction *
SPIRVToLLVMDbgTran::transDebugIntrinsic(const SPIRVExtInst *DebugInst,
                                        BasicBlock *BB) {
  auto GetLocalVar = [&](SPIRVId Id) -> std::pair<DILocalVariable *, DebugLoc> {
    auto *LV = transDebugInst<DILocalVariable>(BM->get<SPIRVExtInst>(Id));
    DebugLoc DL = DebugLoc::get(LV->getLine(), 0, LV->getScope());
    return std::make_pair(LV, DL);
  };
  auto GetValue = [&](SPIRVId Id) -> Value * {
    auto *V = BM->get<SPIRVValue>(Id);
    return SPIRVReader->transValue(V, BB->getParent(), BB);
  };
  auto GetExpression = [&](SPIRVId Id) -> DIExpression * {
    return transDebugInst<DIExpression>(BM->get<SPIRVExtInst>(Id));
  };
  SPIRVWordVec Ops = DebugInst->getArguments();
  switch (DebugInst->getExtOp()) {
  case SPIRVDebug::Scope:
  case SPIRVDebug::NoScope:
    return nullptr;
  case SPIRVDebug::Declare: {
    using namespace SPIRVDebug::Operand::DebugDeclare;
    auto LocalVar = GetLocalVar(Ops[DebugLocalVarIdx]);
    return Builder.insertDeclare(GetValue(Ops[VariableIdx]), LocalVar.first,
                                 GetExpression(Ops[ExpressionIdx]),
                                 LocalVar.second, BB);
  }
  case SPIRVDebug::Value: {
    using namespace SPIRVDebug::Operand::DebugValue;
    auto LocalVar = GetLocalVar(Ops[DebugLocalVarIdx]);
    return Builder.insertDbgValueIntrinsic(
        GetValue(Ops[ValueIdx]), LocalVar.first,
        GetExpression(Ops[ExpressionIdx]), LocalVar.second, BB);
  }
  default:
    llvm_unreachable("Unknown debug intrinsic!");
  }
}

DebugLoc SPIRVToLLVMDbgTran::transDebugScope(const SPIRVInstruction *Inst) {
  unsigned Line = 0;
  unsigned Col = 0;
  MDNode *Scope = nullptr;
  MDNode *InlinedAt = nullptr;
  if (auto L = Inst->getLine()) {
    Line = L->getLine();
    Col = L->getColumn();
  }
  if (SPIRVEntry *S = Inst->getDebugScope()) {
    using namespace SPIRVDebug::Operand::Scope;
    SPIRVExtInst *DbgScope = static_cast<SPIRVExtInst *>(S);
    SPIRVWordVec Ops = DbgScope->getArguments();
    Scope = getScope(BM->getEntry(Ops[ScopeIdx]));
    if (Ops.size() > InlinedAtIdx)
      InlinedAt = transDebugInst(BM->get<SPIRVExtInst>(Ops[InlinedAtIdx]));
  }
  return DebugLoc::get(Line, Col, Scope, InlinedAt);
}

MDNode *SPIRVToLLVMDbgTran::transDebugInlined(const SPIRVExtInst *Inst) {
  using namespace SPIRVDebug::Operand::InlinedAt;
  SPIRVWordVec Ops = Inst->getArguments();
  assert(Ops.size() >= MinOperandCount && "Invalid number of operands");
  unsigned Line = Ops[LineIdx];
  unsigned Col = 0; // DebugInlinedAt instruction has no column operand
  DILocalScope *Scope =
      cast<DILocalScope>(getScope(BM->getEntry(Ops[ScopeIdx])));
  DILocation *InlinedAt = nullptr;
  if (Ops.size() > InlinedIdx) {
    InlinedAt =
        transDebugInst<DILocation>(BM->get<SPIRVExtInst>(Ops[InlinedIdx]));
  }
  return DILocation::getDistinct(M->getContext(), Line, Col, Scope, InlinedAt);
}

void SPIRVToLLVMDbgTran::finalize() {
  if (!Enable)
    return;
  Builder.finalize();
}

DIFile *SPIRVToLLVMDbgTran::getFile(const SPIRVId SourceId) {
  using namespace SPIRVDebug::Operand::Source;
  SPIRVExtInst *Source = BM->get<SPIRVExtInst>(SourceId);
  assert(Source->getExtOp() == SPIRVDebug::Source &&
         "DebugSource instruction is expected");
  SPIRVWordVec SourceArgs = Source->getArguments();
  assert(SourceArgs.size() == OperandCount && "Invalid number of operands");
  return getDIFile(getString(SourceArgs[FileIdx]));
}

SPIRVToLLVMDbgTran::SplitFileName::SplitFileName(const string &FileName) {
  auto Loc = FileName.find_last_of("/\\");
  if (Loc != std::string::npos) {
    BaseName = FileName.substr(Loc + 1);
    Path = FileName.substr(0, Loc);
  } else {
    BaseName = FileName;
    Path = ".";
  }
}

} // namespace SPIRV
