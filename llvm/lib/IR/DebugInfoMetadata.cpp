//===- DebugInfoMetadata.cpp - Implement debug info metadata --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the debug info Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "LLVMContextImpl.h"
#include "MetadataImpl.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"

using namespace llvm;

DILocation::DILocation(LLVMContext &C, StorageType Storage, unsigned Line,
                       unsigned Column, ArrayRef<Metadata *> MDs)
    : MDNode(C, DILocationKind, Storage, MDs) {
  assert((MDs.size() == 1 || MDs.size() == 2) &&
         "Expected a scope and optional inlined-at");

  // Set line and column.
  assert(Column < (1u << 16) && "Expected 16-bit column");

  SubclassData32 = Line;
  SubclassData16 = Column;
}

static void adjustColumn(unsigned &Column) {
  // Set to unknown on overflow.  We only have 16 bits to play with here.
  if (Column >= (1u << 16))
    Column = 0;
}

DILocation *DILocation::getImpl(LLVMContext &Context, unsigned Line,
                                unsigned Column, Metadata *Scope,
                                Metadata *InlinedAt, StorageType Storage,
                                bool ShouldCreate) {
  // Fixup column.
  adjustColumn(Column);

  if (Storage == Uniqued) {
    if (auto *N =
            getUniqued(Context.pImpl->DILocations,
                       DILocationInfo::KeyTy(Line, Column, Scope, InlinedAt)))
      return N;
    if (!ShouldCreate)
      return nullptr;
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  SmallVector<Metadata *, 2> Ops;
  Ops.push_back(Scope);
  if (InlinedAt)
    Ops.push_back(InlinedAt);
  return storeImpl(new (Ops.size())
                       DILocation(Context, Storage, Line, Column, Ops),
                   Storage, Context.pImpl->DILocations);
}

DINode::DIFlags DINode::getFlag(StringRef Flag) {
  return StringSwitch<DIFlags>(Flag)
#define HANDLE_DI_FLAG(ID, NAME) .Case("DIFlag" #NAME, Flag##NAME)
#include "llvm/IR/DebugInfoFlags.def"
      .Default(DINode::FlagZero);
}

StringRef DINode::getFlagString(DIFlags Flag) {
  switch (Flag) {
#define HANDLE_DI_FLAG(ID, NAME)                                               \
  case Flag##NAME:                                                             \
    return "DIFlag" #NAME;
#include "llvm/IR/DebugInfoFlags.def"
  }
  return "";
}

DINode::DIFlags DINode::splitFlags(DIFlags Flags,
                                   SmallVectorImpl<DIFlags> &SplitFlags) {
  // Flags that are packed together need to be specially handled, so
  // that, for example, we emit "DIFlagPublic" and not
  // "DIFlagPrivate | DIFlagProtected".
  if (DIFlags A = Flags & FlagAccessibility) {
    if (A == FlagPrivate)
      SplitFlags.push_back(FlagPrivate);
    else if (A == FlagProtected)
      SplitFlags.push_back(FlagProtected);
    else
      SplitFlags.push_back(FlagPublic);
    Flags &= ~A;
  }
  if (DIFlags R = Flags & FlagPtrToMemberRep) {
    if (R == FlagSingleInheritance)
      SplitFlags.push_back(FlagSingleInheritance);
    else if (R == FlagMultipleInheritance)
      SplitFlags.push_back(FlagMultipleInheritance);
    else
      SplitFlags.push_back(FlagVirtualInheritance);
    Flags &= ~R;
  }
  if ((Flags & FlagIndirectVirtualBase) == FlagIndirectVirtualBase) {
    Flags &= ~FlagIndirectVirtualBase;
    SplitFlags.push_back(FlagIndirectVirtualBase);
  }

#define HANDLE_DI_FLAG(ID, NAME)                                               \
  if (DIFlags Bit = Flags & Flag##NAME) {                                      \
    SplitFlags.push_back(Bit);                                                 \
    Flags &= ~Bit;                                                             \
  }
#include "llvm/IR/DebugInfoFlags.def"
  return Flags;
}

DIScopeRef DIScope::getScope() const {
  if (auto *T = dyn_cast<DIType>(this))
    return T->getScope();

  if (auto *SP = dyn_cast<DISubprogram>(this))
    return SP->getScope();

  if (auto *LB = dyn_cast<DILexicalBlockBase>(this))
    return LB->getScope();

  if (auto *NS = dyn_cast<DINamespace>(this))
    return NS->getScope();

  if (auto *M = dyn_cast<DIModule>(this))
    return M->getScope();

  assert((isa<DIFile>(this) || isa<DICompileUnit>(this)) &&
         "Unhandled type of scope.");
  return nullptr;
}

StringRef DIScope::getName() const {
  if (auto *T = dyn_cast<DIType>(this))
    return T->getName();
  if (auto *SP = dyn_cast<DISubprogram>(this))
    return SP->getName();
  if (auto *NS = dyn_cast<DINamespace>(this))
    return NS->getName();
  if (auto *M = dyn_cast<DIModule>(this))
    return M->getName();
  assert((isa<DILexicalBlockBase>(this) || isa<DIFile>(this) ||
          isa<DICompileUnit>(this)) &&
         "Unhandled type of scope.");
  return "";
}

#ifndef NDEBUG
static bool isCanonical(const MDString *S) {
  return !S || !S->getString().empty();
}
#endif

GenericDINode *GenericDINode::getImpl(LLVMContext &Context, unsigned Tag,
                                      MDString *Header,
                                      ArrayRef<Metadata *> DwarfOps,
                                      StorageType Storage, bool ShouldCreate) {
  unsigned Hash = 0;
  if (Storage == Uniqued) {
    GenericDINodeInfo::KeyTy Key(Tag, Header, DwarfOps);
    if (auto *N = getUniqued(Context.pImpl->GenericDINodes, Key))
      return N;
    if (!ShouldCreate)
      return nullptr;
    Hash = Key.getHash();
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  // Use a nullptr for empty headers.
  assert(isCanonical(Header) && "Expected canonical MDString");
  Metadata *PreOps[] = {Header};
  return storeImpl(new (DwarfOps.size() + 1) GenericDINode(
                       Context, Storage, Hash, Tag, PreOps, DwarfOps),
                   Storage, Context.pImpl->GenericDINodes);
}

void GenericDINode::recalculateHash() {
  setHash(GenericDINodeInfo::KeyTy::calculateHash(this));
}

#define UNWRAP_ARGS_IMPL(...) __VA_ARGS__
#define UNWRAP_ARGS(ARGS) UNWRAP_ARGS_IMPL ARGS
#define DEFINE_GETIMPL_LOOKUP(CLASS, ARGS)                                     \
  do {                                                                         \
    if (Storage == Uniqued) {                                                  \
      if (auto *N = getUniqued(Context.pImpl->CLASS##s,                        \
                               CLASS##Info::KeyTy(UNWRAP_ARGS(ARGS))))         \
        return N;                                                              \
      if (!ShouldCreate)                                                       \
        return nullptr;                                                        \
    } else {                                                                   \
      assert(ShouldCreate &&                                                   \
             "Expected non-uniqued nodes to always be created");               \
    }                                                                          \
  } while (false)
#define DEFINE_GETIMPL_STORE(CLASS, ARGS, OPS)                                 \
  return storeImpl(new (array_lengthof(OPS))                                   \
                       CLASS(Context, Storage, UNWRAP_ARGS(ARGS), OPS),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_OPS(CLASS, ARGS)                               \
  return storeImpl(new (0u) CLASS(Context, Storage, UNWRAP_ARGS(ARGS)),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(CLASS, OPS)                   \
  return storeImpl(new (array_lengthof(OPS)) CLASS(Context, Storage, OPS),     \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_N(CLASS, ARGS, OPS, NUM_OPS)                      \
  return storeImpl(new (NUM_OPS)                                               \
                       CLASS(Context, Storage, UNWRAP_ARGS(ARGS), OPS),        \
                   Storage, Context.pImpl->CLASS##s)

DISubrange *DISubrange::getImpl(LLVMContext &Context, int64_t Count, int64_t Lo,
                                StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DISubrange, (Count, Lo));
  DEFINE_GETIMPL_STORE_NO_OPS(DISubrange, (Count, Lo));
}

DIEnumerator *DIEnumerator::getImpl(LLVMContext &Context, int64_t Value,
                                    MDString *Name, StorageType Storage,
                                    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIEnumerator, (Value, Name));
  Metadata *Ops[] = {Name};
  DEFINE_GETIMPL_STORE(DIEnumerator, (Value), Ops);
}

DIBasicType *DIBasicType::getImpl(LLVMContext &Context, unsigned Tag,
                                  MDString *Name, uint64_t SizeInBits,
                                  uint32_t AlignInBits, unsigned Encoding,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIBasicType,
                        (Tag, Name, SizeInBits, AlignInBits, Encoding));
  Metadata *Ops[] = {nullptr, nullptr, Name};
  DEFINE_GETIMPL_STORE(DIBasicType, (Tag, SizeInBits, AlignInBits, Encoding),
                       Ops);
}

DIDerivedType *DIDerivedType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits,
    Optional<unsigned> DWARFAddressSpace, DIFlags Flags, Metadata *ExtraData,
    StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIDerivedType,
                        (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                         AlignInBits, OffsetInBits, DWARFAddressSpace, Flags,
                         ExtraData));
  Metadata *Ops[] = {File, Scope, Name, BaseType, ExtraData};
  DEFINE_GETIMPL_STORE(
      DIDerivedType, (Tag, Line, SizeInBits, AlignInBits, OffsetInBits,
                      DWARFAddressSpace, Flags), Ops);
}

DICompositeType *DICompositeType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, DIFlags Flags,
    Metadata *Elements, unsigned RuntimeLang, Metadata *VTableHolder,
    Metadata *TemplateParams, MDString *Identifier, StorageType Storage,
    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");

  // Keep this in sync with buildODRType.
  DEFINE_GETIMPL_LOOKUP(
      DICompositeType, (Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                        AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                        VTableHolder, TemplateParams, Identifier));
  Metadata *Ops[] = {File,     Scope,        Name,           BaseType,
                     Elements, VTableHolder, TemplateParams, Identifier};
  DEFINE_GETIMPL_STORE(DICompositeType, (Tag, Line, RuntimeLang, SizeInBits,
                                         AlignInBits, OffsetInBits, Flags),
                       Ops);
}

DICompositeType *DICompositeType::buildODRType(
    LLVMContext &Context, MDString &Identifier, unsigned Tag, MDString *Name,
    Metadata *File, unsigned Line, Metadata *Scope, Metadata *BaseType,
    uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits,
    DIFlags Flags, Metadata *Elements, unsigned RuntimeLang,
    Metadata *VTableHolder, Metadata *TemplateParams) {
  assert(!Identifier.getString().empty() && "Expected valid identifier");
  if (!Context.isODRUniquingDebugTypes())
    return nullptr;
  auto *&CT = (*Context.pImpl->DITypeMap)[&Identifier];
  if (!CT)
    return CT = DICompositeType::getDistinct(
               Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
               AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
               VTableHolder, TemplateParams, &Identifier);

  // Only mutate CT if it's a forward declaration and the new operands aren't.
  assert(CT->getRawIdentifier() == &Identifier && "Wrong ODR identifier?");
  if (!CT->isForwardDecl() || (Flags & DINode::FlagFwdDecl))
    return CT;

  // Mutate CT in place.  Keep this in sync with getImpl.
  CT->mutate(Tag, Line, RuntimeLang, SizeInBits, AlignInBits, OffsetInBits,
             Flags);
  Metadata *Ops[] = {File,     Scope,        Name,           BaseType,
                     Elements, VTableHolder, TemplateParams, &Identifier};
  assert((std::end(Ops) - std::begin(Ops)) == (int)CT->getNumOperands() &&
         "Mismatched number of operands");
  for (unsigned I = 0, E = CT->getNumOperands(); I != E; ++I)
    if (Ops[I] != CT->getOperand(I))
      CT->setOperand(I, Ops[I]);
  return CT;
}

DICompositeType *DICompositeType::getODRType(
    LLVMContext &Context, MDString &Identifier, unsigned Tag, MDString *Name,
    Metadata *File, unsigned Line, Metadata *Scope, Metadata *BaseType,
    uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits,
    DIFlags Flags, Metadata *Elements, unsigned RuntimeLang,
    Metadata *VTableHolder, Metadata *TemplateParams) {
  assert(!Identifier.getString().empty() && "Expected valid identifier");
  if (!Context.isODRUniquingDebugTypes())
    return nullptr;
  auto *&CT = (*Context.pImpl->DITypeMap)[&Identifier];
  if (!CT)
    CT = DICompositeType::getDistinct(
        Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
        AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang, VTableHolder,
        TemplateParams, &Identifier);
  return CT;
}

DICompositeType *DICompositeType::getODRTypeIfExists(LLVMContext &Context,
                                                     MDString &Identifier) {
  assert(!Identifier.getString().empty() && "Expected valid identifier");
  if (!Context.isODRUniquingDebugTypes())
    return nullptr;
  return Context.pImpl->DITypeMap->lookup(&Identifier);
}

DISubroutineType *DISubroutineType::getImpl(LLVMContext &Context, DIFlags Flags,
                                            uint8_t CC, Metadata *TypeArray,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DISubroutineType, (Flags, CC, TypeArray));
  Metadata *Ops[] = {nullptr, nullptr, nullptr, TypeArray};
  DEFINE_GETIMPL_STORE(DISubroutineType, (Flags, CC), Ops);
}

// FIXME: Implement this string-enum correspondence with a .def file and macros,
// so that the association is explicit rather than implied.
static const char *ChecksumKindName[DIFile::CSK_Last + 1] = {
  "CSK_None",
  "CSK_MD5",
  "CSK_SHA1"
};

DIFile::ChecksumKind DIFile::getChecksumKind(StringRef CSKindStr) {
  return StringSwitch<DIFile::ChecksumKind>(CSKindStr)
      .Case("CSK_MD5", DIFile::CSK_MD5)
      .Case("CSK_SHA1", DIFile::CSK_SHA1)
      .Default(DIFile::CSK_None);
}

StringRef DIFile::getChecksumKindAsString() const {
  assert(CSKind <= DIFile::CSK_Last && "Invalid checksum kind");
  return ChecksumKindName[CSKind];
}

DIFile *DIFile::getImpl(LLVMContext &Context, MDString *Filename,
                        MDString *Directory, DIFile::ChecksumKind CSKind,
                        MDString *Checksum, StorageType Storage,
                        bool ShouldCreate) {
  assert(isCanonical(Filename) && "Expected canonical MDString");
  assert(isCanonical(Directory) && "Expected canonical MDString");
  assert(isCanonical(Checksum) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIFile, (Filename, Directory, CSKind, Checksum));
  Metadata *Ops[] = {Filename, Directory, Checksum};
  DEFINE_GETIMPL_STORE(DIFile, (CSKind), Ops);
}

DICompileUnit *DICompileUnit::getImpl(
    LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
    MDString *Producer, bool IsOptimized, MDString *Flags,
    unsigned RuntimeVersion, MDString *SplitDebugFilename,
    unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
    Metadata *GlobalVariables, Metadata *ImportedEntities, Metadata *Macros,
    uint64_t DWOId, bool SplitDebugInlining, bool DebugInfoForProfiling,
    bool GnuPubnames, StorageType Storage, bool ShouldCreate) {
  assert(Storage != Uniqued && "Cannot unique DICompileUnit");
  assert(isCanonical(Producer) && "Expected canonical MDString");
  assert(isCanonical(Flags) && "Expected canonical MDString");
  assert(isCanonical(SplitDebugFilename) && "Expected canonical MDString");

  Metadata *Ops[] = {
      File,      Producer,      Flags,           SplitDebugFilename,
      EnumTypes, RetainedTypes, GlobalVariables, ImportedEntities,
      Macros};
  return storeImpl(new (array_lengthof(Ops)) DICompileUnit(
                       Context, Storage, SourceLanguage, IsOptimized,
                       RuntimeVersion, EmissionKind, DWOId, SplitDebugInlining,
                       DebugInfoForProfiling, GnuPubnames, Ops),
                   Storage);
}

Optional<DICompileUnit::DebugEmissionKind>
DICompileUnit::getEmissionKind(StringRef Str) {
  return StringSwitch<Optional<DebugEmissionKind>>(Str)
      .Case("NoDebug", NoDebug)
      .Case("FullDebug", FullDebug)
      .Case("LineTablesOnly", LineTablesOnly)
      .Default(None);
}

const char *DICompileUnit::EmissionKindString(DebugEmissionKind EK) {
  switch (EK) {
  case NoDebug:        return "NoDebug";
  case FullDebug:      return "FullDebug";
  case LineTablesOnly: return "LineTablesOnly";
  }
  return nullptr;
}

DISubprogram *DILocalScope::getSubprogram() const {
  if (auto *Block = dyn_cast<DILexicalBlockBase>(this))
    return Block->getScope()->getSubprogram();
  return const_cast<DISubprogram *>(cast<DISubprogram>(this));
}

DILocalScope *DILocalScope::getNonLexicalBlockFileScope() const {
  if (auto *File = dyn_cast<DILexicalBlockFile>(this))
    return File->getScope()->getNonLexicalBlockFileScope();
  return const_cast<DILocalScope *>(this);
}

DISubprogram *DISubprogram::getImpl(
    LLVMContext &Context, Metadata *Scope, MDString *Name,
    MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
    bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
    Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
    int ThisAdjustment, DIFlags Flags, bool IsOptimized, Metadata *Unit,
    Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
    Metadata *ThrownTypes, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      DISubprogram, (Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
                     IsDefinition, ScopeLine, ContainingType, Virtuality,
                     VirtualIndex, ThisAdjustment, Flags, IsOptimized, Unit,
                     TemplateParams, Declaration, Variables, ThrownTypes));
  SmallVector<Metadata *, 11> Ops = {
      File,        Scope,     Name,           LinkageName,    Type,       Unit,
      Declaration, Variables, ContainingType, TemplateParams, ThrownTypes};
  if (!ThrownTypes) {
    Ops.pop_back();
    if (!TemplateParams) {
      Ops.pop_back();
      if (!ContainingType)
        Ops.pop_back();
    }
  }
  DEFINE_GETIMPL_STORE_N(DISubprogram,
                         (Line, ScopeLine, Virtuality, VirtualIndex,
                          ThisAdjustment, Flags, IsLocalToUnit, IsDefinition,
                          IsOptimized),
                         Ops, Ops.size());
}

bool DISubprogram::describes(const Function *F) const {
  assert(F && "Invalid function");
  if (F->getSubprogram() == this)
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  return F->getName() == Name;
}

DILexicalBlock *DILexicalBlock::getImpl(LLVMContext &Context, Metadata *Scope,
                                        Metadata *File, unsigned Line,
                                        unsigned Column, StorageType Storage,
                                        bool ShouldCreate) {
  // Fixup column.
  adjustColumn(Column);

  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(DILexicalBlock, (Scope, File, Line, Column));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(DILexicalBlock, (Line, Column), Ops);
}

DILexicalBlockFile *DILexicalBlockFile::getImpl(LLVMContext &Context,
                                                Metadata *Scope, Metadata *File,
                                                unsigned Discriminator,
                                                StorageType Storage,
                                                bool ShouldCreate) {
  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(DILexicalBlockFile, (Scope, File, Discriminator));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(DILexicalBlockFile, (Discriminator), Ops);
}

DINamespace *DINamespace::getImpl(LLVMContext &Context, Metadata *Scope,
                                  MDString *Name, bool ExportSymbols,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DINamespace, (Scope, Name, ExportSymbols));
  // The nullptr is for DIScope's File operand. This should be refactored.
  Metadata *Ops[] = {nullptr, Scope, Name};
  DEFINE_GETIMPL_STORE(DINamespace, (ExportSymbols), Ops);
}

DIModule *DIModule::getImpl(LLVMContext &Context, Metadata *Scope,
                            MDString *Name, MDString *ConfigurationMacros,
                            MDString *IncludePath, MDString *ISysRoot,
                            StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      DIModule, (Scope, Name, ConfigurationMacros, IncludePath, ISysRoot));
  Metadata *Ops[] = {Scope, Name, ConfigurationMacros, IncludePath, ISysRoot};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DIModule, Ops);
}

DITemplateTypeParameter *DITemplateTypeParameter::getImpl(LLVMContext &Context,
                                                          MDString *Name,
                                                          Metadata *Type,
                                                          StorageType Storage,
                                                          bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DITemplateTypeParameter, (Name, Type));
  Metadata *Ops[] = {Name, Type};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DITemplateTypeParameter, Ops);
}

DITemplateValueParameter *DITemplateValueParameter::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *Type,
    Metadata *Value, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DITemplateValueParameter, (Tag, Name, Type, Value));
  Metadata *Ops[] = {Name, Type, Value};
  DEFINE_GETIMPL_STORE(DITemplateValueParameter, (Tag), Ops);
}

DIGlobalVariable *
DIGlobalVariable::getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
                          MDString *LinkageName, Metadata *File, unsigned Line,
                          Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
                          Metadata *StaticDataMemberDeclaration,
                          uint32_t AlignInBits, StorageType Storage,
                          bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIGlobalVariable,
                        (Scope, Name, LinkageName, File, Line, Type,
                         IsLocalToUnit, IsDefinition,
                         StaticDataMemberDeclaration, AlignInBits));
  Metadata *Ops[] = {
      Scope, Name, File, Type, Name, LinkageName, StaticDataMemberDeclaration};
  DEFINE_GETIMPL_STORE(DIGlobalVariable,
                       (Line, IsLocalToUnit, IsDefinition, AlignInBits),
                       Ops);
}

DILocalVariable *DILocalVariable::getImpl(LLVMContext &Context, Metadata *Scope,
                                          MDString *Name, Metadata *File,
                                          unsigned Line, Metadata *Type,
                                          unsigned Arg, DIFlags Flags,
                                          uint32_t AlignInBits,
                                          StorageType Storage,
                                          bool ShouldCreate) {
  // 64K ought to be enough for any frontend.
  assert(Arg <= UINT16_MAX && "Expected argument number to fit in 16-bits");

  assert(Scope && "Expected scope");
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DILocalVariable,
                        (Scope, Name, File, Line, Type, Arg, Flags,
                         AlignInBits));
  Metadata *Ops[] = {Scope, Name, File, Type};
  DEFINE_GETIMPL_STORE(DILocalVariable, (Line, Arg, Flags, AlignInBits), Ops);
}

DIExpression *DIExpression::getImpl(LLVMContext &Context,
                                    ArrayRef<uint64_t> Elements,
                                    StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DIExpression, (Elements));
  DEFINE_GETIMPL_STORE_NO_OPS(DIExpression, (Elements));
}

unsigned DIExpression::ExprOperand::getSize() const {
  switch (getOp()) {
  case dwarf::DW_OP_LLVM_fragment:
    return 3;
  case dwarf::DW_OP_constu:
  case dwarf::DW_OP_plus_uconst:
    return 2;
  default:
    return 1;
  }
}

bool DIExpression::isValid() const {
  for (auto I = expr_op_begin(), E = expr_op_end(); I != E; ++I) {
    // Check that there's space for the operand.
    if (I->get() + I->getSize() > E->get())
      return false;

    // Check that the operand is valid.
    switch (I->getOp()) {
    default:
      return false;
    case dwarf::DW_OP_LLVM_fragment:
      // A fragment operator must appear at the end.
      return I->get() + I->getSize() == E->get();
    case dwarf::DW_OP_stack_value: {
      // Must be the last one or followed by a DW_OP_LLVM_fragment.
      if (I->get() + I->getSize() == E->get())
        break;
      auto J = I;
      if ((++J)->getOp() != dwarf::DW_OP_LLVM_fragment)
        return false;
      break;
    }
    case dwarf::DW_OP_swap: {
      // Must be more than one implicit element on the stack.

      // FIXME: A better way to implement this would be to add a local variable
      // that keeps track of the stack depth and introduce something like a
      // DW_LLVM_OP_implicit_location as a placeholder for the location this
      // DIExpression is attached to, or else pass the number of implicit stack
      // elements into isValid.
      if (getNumElements() == 1)
        return false;
      break;
    }
    case dwarf::DW_OP_constu:
    case dwarf::DW_OP_plus_uconst:
    case dwarf::DW_OP_plus:
    case dwarf::DW_OP_minus:
    case dwarf::DW_OP_deref:
    case dwarf::DW_OP_xderef:
      break;
    }
  }
  return true;
}

Optional<DIExpression::FragmentInfo>
DIExpression::getFragmentInfo(expr_op_iterator Start, expr_op_iterator End) {
  for (auto I = Start; I != End; ++I)
    if (I->getOp() == dwarf::DW_OP_LLVM_fragment) {
      DIExpression::FragmentInfo Info = {I->getArg(1), I->getArg(0)};
      return Info;
    }
  return None;
}

void DIExpression::appendOffset(SmallVectorImpl<uint64_t> &Ops,
                                int64_t Offset) {
  if (Offset > 0) {
    Ops.push_back(dwarf::DW_OP_plus_uconst);
    Ops.push_back(Offset);
  } else if (Offset < 0) {
    Ops.push_back(dwarf::DW_OP_constu);
    Ops.push_back(-Offset);
    Ops.push_back(dwarf::DW_OP_minus);
  }
}

bool DIExpression::extractIfOffset(int64_t &Offset) const {
  if (getNumElements() == 0) {
    Offset = 0;
    return true;
  }

  if (getNumElements() == 2 && Elements[0] == dwarf::DW_OP_plus_uconst) {
    Offset = Elements[1];
    return true;
  }

  if (getNumElements() == 3 && Elements[0] == dwarf::DW_OP_constu) {
    if (Elements[2] == dwarf::DW_OP_plus) {
      Offset = Elements[1];
      return true;
    }
    if (Elements[2] == dwarf::DW_OP_minus) {
      Offset = -Elements[1];
      return true;
    }
  }

  return false;
}

DIExpression *DIExpression::prepend(const DIExpression *Expr, bool Deref,
                                    int64_t Offset, bool StackValue) {
  SmallVector<uint64_t, 8> Ops;
  appendOffset(Ops, Offset);
  if (Deref)
    Ops.push_back(dwarf::DW_OP_deref);
  if (Expr)
    for (auto Op : Expr->expr_ops()) {
      // A DW_OP_stack_value comes at the end, but before a DW_OP_LLVM_fragment.
      if (StackValue) {
        if (Op.getOp() == dwarf::DW_OP_stack_value)
          StackValue = false;
        else if (Op.getOp() == dwarf::DW_OP_LLVM_fragment) {
          Ops.push_back(dwarf::DW_OP_stack_value);
          StackValue = false;
        }
      }
      Ops.push_back(Op.getOp());
      for (unsigned I = 0; I < Op.getNumArgs(); ++I)
        Ops.push_back(Op.getArg(I));
    }
  if (StackValue)
    Ops.push_back(dwarf::DW_OP_stack_value);
  return DIExpression::get(Expr->getContext(), Ops);
}

DIExpression *DIExpression::createFragmentExpression(const DIExpression *Expr,
                                                     unsigned OffsetInBits,
                                                     unsigned SizeInBits) {
  SmallVector<uint64_t, 8> Ops;
  // Copy over the expression, but leave off any trailing DW_OP_LLVM_fragment.
  if (Expr) {
    for (auto Op : Expr->expr_ops()) {
      if (Op.getOp() == dwarf::DW_OP_LLVM_fragment) {
        // Make the new offset point into the existing fragment.
        uint64_t FragmentOffsetInBits = Op.getArg(0);
        // Op.getArg(0) is FragmentOffsetInBits.
        // Op.getArg(1) is FragmentSizeInBits.
        assert((OffsetInBits + SizeInBits <= Op.getArg(0) + Op.getArg(1)) &&
               "new fragment outside of original fragment");
        OffsetInBits += FragmentOffsetInBits;
        break;
      }
      Ops.push_back(Op.getOp());
      for (unsigned I = 0; I < Op.getNumArgs(); ++I)
        Ops.push_back(Op.getArg(I));
    }
  }
  Ops.push_back(dwarf::DW_OP_LLVM_fragment);
  Ops.push_back(OffsetInBits);
  Ops.push_back(SizeInBits);
  return DIExpression::get(Expr->getContext(), Ops);
}

bool DIExpression::isConstant() const {
  // Recognize DW_OP_constu C DW_OP_stack_value (DW_OP_LLVM_fragment Len Ofs)?.
  if (getNumElements() != 3 && getNumElements() != 6)
    return false;
  if (getElement(0) != dwarf::DW_OP_constu ||
      getElement(2) != dwarf::DW_OP_stack_value)
    return false;
  if (getNumElements() == 6 && getElement(3) != dwarf::DW_OP_LLVM_fragment)
    return false;
  return true;
}

DIGlobalVariableExpression *
DIGlobalVariableExpression::getImpl(LLVMContext &Context, Metadata *Variable,
                                    Metadata *Expression, StorageType Storage,
                                    bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DIGlobalVariableExpression, (Variable, Expression));
  Metadata *Ops[] = {Variable, Expression};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DIGlobalVariableExpression, Ops);
}

DIObjCProperty *DIObjCProperty::getImpl(
    LLVMContext &Context, MDString *Name, Metadata *File, unsigned Line,
    MDString *GetterName, MDString *SetterName, unsigned Attributes,
    Metadata *Type, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(GetterName) && "Expected canonical MDString");
  assert(isCanonical(SetterName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIObjCProperty, (Name, File, Line, GetterName,
                                         SetterName, Attributes, Type));
  Metadata *Ops[] = {Name, File, GetterName, SetterName, Type};
  DEFINE_GETIMPL_STORE(DIObjCProperty, (Line, Attributes), Ops);
}

DIImportedEntity *DIImportedEntity::getImpl(LLVMContext &Context, unsigned Tag,
                                            Metadata *Scope, Metadata *Entity,
                                            Metadata *File, unsigned Line,
                                            MDString *Name, StorageType Storage,
                                            bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIImportedEntity,
                        (Tag, Scope, Entity, File, Line, Name));
  Metadata *Ops[] = {Scope, Entity, Name, File};
  DEFINE_GETIMPL_STORE(DIImportedEntity, (Tag, Line), Ops);
}

DIMacro *DIMacro::getImpl(LLVMContext &Context, unsigned MIType,
                          unsigned Line, MDString *Name, MDString *Value,
                          StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIMacro, (MIType, Line, Name, Value));
  Metadata *Ops[] = { Name, Value };
  DEFINE_GETIMPL_STORE(DIMacro, (MIType, Line), Ops);
}

DIMacroFile *DIMacroFile::getImpl(LLVMContext &Context, unsigned MIType,
                                  unsigned Line, Metadata *File,
                                  Metadata *Elements, StorageType Storage,
                                  bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DIMacroFile,
                        (MIType, Line, File, Elements));
  Metadata *Ops[] = { File, Elements };
  DEFINE_GETIMPL_STORE(DIMacroFile, (MIType, Line), Ops);
}

