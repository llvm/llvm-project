//===--- DIBuilder.cpp - Debug Information Builder ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DIBuilder.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DIBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "LLVMContextImpl.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {
class HeaderBuilder {
  /// \brief Whether there are any fields yet.
  ///
  /// Note that this is not equivalent to \c Chars.empty(), since \a concat()
  /// may have been called already with an empty string.
  bool IsEmpty;
  SmallVector<char, 256> Chars;

public:
  HeaderBuilder() : IsEmpty(true) {}
  HeaderBuilder(const HeaderBuilder &X) : IsEmpty(X.IsEmpty), Chars(X.Chars) {}
  HeaderBuilder(HeaderBuilder &&X)
      : IsEmpty(X.IsEmpty), Chars(std::move(X.Chars)) {}

  template <class Twineable> HeaderBuilder &concat(Twineable &&X) {
    if (IsEmpty)
      IsEmpty = false;
    else
      Chars.push_back(0);
    Twine(X).toVector(Chars);
    return *this;
  }

  MDString *get(LLVMContext &Context) const {
    return MDString::get(Context, StringRef(Chars.begin(), Chars.size()));
  }

  static HeaderBuilder get(unsigned Tag) {
    return HeaderBuilder().concat("0x" + Twine::utohexstr(Tag));
  }
};
}

DIBuilder::DIBuilder(Module &m, bool AllowUnresolvedNodes)
  : M(m), VMContext(M.getContext()), CUNode(nullptr),
      DeclareFn(nullptr), ValueFn(nullptr),
      AllowUnresolvedNodes(AllowUnresolvedNodes) {}

void DIBuilder::trackIfUnresolved(MDNode *N) {
  if (!N)
    return;
  if (N->isResolved())
    return;

  assert(AllowUnresolvedNodes && "Cannot handle unresolved nodes");
  UnresolvedNodes.emplace_back(N);
}

void DIBuilder::finalize() {
  if (!CUNode) {
    assert(!AllowUnresolvedNodes &&
           "creating type nodes without a CU is not supported");
    return;
  }

  CUNode->replaceEnumTypes(MDTuple::get(VMContext, AllEnumTypes));

  SmallVector<Metadata *, 16> RetainValues;
  // Declarations and definitions of the same type may be retained. Some
  // clients RAUW these pairs, leaving duplicates in the retained types
  // list. Use a set to remove the duplicates while we transform the
  // TrackingVHs back into Values.
  SmallPtrSet<Metadata *, 16> RetainSet;
  for (unsigned I = 0, E = AllRetainTypes.size(); I < E; I++)
    if (RetainSet.insert(AllRetainTypes[I]).second)
      RetainValues.push_back(AllRetainTypes[I]);

  if (!RetainValues.empty())
    CUNode->replaceRetainedTypes(MDTuple::get(VMContext, RetainValues));

  DISubprogramArray SPs = MDTuple::get(VMContext, AllSubprograms);
  auto resolveVariables = [&](DISubprogram *SP) {
    MDTuple *Temp = SP->getVariables().get();
    if (!Temp)
      return;

    SmallVector<Metadata *, 4> Variables;

    auto PV = PreservedVariables.find(SP);
    if (PV != PreservedVariables.end())
      Variables.append(PV->second.begin(), PV->second.end());

    DINodeArray AV = getOrCreateArray(Variables);
    TempMDTuple(Temp)->replaceAllUsesWith(AV.get());
  };
  for (auto *SP : SPs)
    resolveVariables(SP);
  for (auto *N : RetainValues)
    if (auto *SP = dyn_cast<DISubprogram>(N))
      resolveVariables(SP);

  if (!AllGVs.empty())
    CUNode->replaceGlobalVariables(MDTuple::get(VMContext, AllGVs));

  if (!AllImportedModules.empty())
    CUNode->replaceImportedEntities(MDTuple::get(
        VMContext, SmallVector<Metadata *, 16>(AllImportedModules.begin(),
                                               AllImportedModules.end())));

  // Now that all temp nodes have been replaced or deleted, resolve remaining
  // cycles.
  for (const auto &N : UnresolvedNodes)
    if (N && !N->isResolved())
      N->resolveCycles();
  UnresolvedNodes.clear();

  // Can't handle unresolved nodes anymore.
  AllowUnresolvedNodes = false;
}

/// If N is compile unit return NULL otherwise return N.
static DIScope *getNonCompileUnitScope(DIScope *N) {
  if (!N || isa<DICompileUnit>(N))
    return nullptr;
  return cast<DIScope>(N);
}

DICompileUnit *DIBuilder::createCompileUnit(
    unsigned Lang, StringRef Filename, StringRef Directory, StringRef Producer,
    bool isOptimized, StringRef Flags, unsigned RunTimeVer, StringRef SplitName,
    DICompileUnit::DebugEmissionKind Kind, uint64_t DWOId) {

  assert(((Lang <= dwarf::DW_LANG_Fortran08 && Lang >= dwarf::DW_LANG_C89) ||
          (Lang <= dwarf::DW_LANG_hi_user && Lang >= dwarf::DW_LANG_lo_user)) &&
         "Invalid Language tag");
  assert(!Filename.empty() &&
         "Unable to create compile unit without filename");

  assert(!CUNode && "Can only make one compile unit per DIBuilder instance");
  CUNode = DICompileUnit::getDistinct(
      VMContext, Lang, DIFile::get(VMContext, Filename, Directory), Producer,
      isOptimized, Flags, RunTimeVer, SplitName, Kind, nullptr, nullptr,
      nullptr, nullptr, nullptr, DWOId);

  // Create a named metadata so that it is easier to find cu in a module.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
  NMD->addOperand(CUNode);
  trackIfUnresolved(CUNode);
  return CUNode;
}

static DIImportedEntity *
createImportedModule(LLVMContext &C, dwarf::Tag Tag, DIScope *Context,
                     Metadata *NS, unsigned Line, StringRef Name,
                     SmallVectorImpl<TrackingMDNodeRef> &AllImportedModules) {
  unsigned EntitiesCount = C.pImpl->DIImportedEntitys.size();
  auto *M = DIImportedEntity::get(C, Tag, Context, DINodeRef(NS), Line, Name);
  if (EntitiesCount < C.pImpl->DIImportedEntitys.size())
    // A new Imported Entity was just added to the context.
    // Add it to the Imported Modules list.
    AllImportedModules.emplace_back(M);
  return M;
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context,
                                                  DINamespace *NS,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context,
                                                  DIImportedEntity *NS,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context, DIModule *M,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, M, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedDeclaration(DIScope *Context,
                                                       DINode *Decl,
                                                       unsigned Line,
                                                       StringRef Name) {
  // Make sure to use the unique identifier based metadata reference for
  // types that have one.
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_declaration,
                                Context, Decl, Line, Name, AllImportedModules);
}

DIFile *DIBuilder::createFile(StringRef Filename, StringRef Directory) {
  return DIFile::get(VMContext, Filename, Directory);
}

DIEnumerator *DIBuilder::createEnumerator(StringRef Name, int64_t Val) {
  assert(!Name.empty() && "Unable to create enumerator without name");
  return DIEnumerator::get(VMContext, Val, Name);
}

DIBasicType *DIBuilder::createUnspecifiedType(StringRef Name) {
  assert(!Name.empty() && "Unable to create type without name");
  return DIBasicType::get(VMContext, dwarf::DW_TAG_unspecified_type, Name);
}

DIBasicType *DIBuilder::createNullPtrType() {
  return createUnspecifiedType("decltype(nullptr)");
}

DIBasicType *DIBuilder::createBasicType(StringRef Name, uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        unsigned Encoding) {
  assert(!Name.empty() && "Unable to create type without name");
  return DIBasicType::get(VMContext, dwarf::DW_TAG_base_type, Name, SizeInBits,
                          AlignInBits, Encoding);
}

DIDerivedType *DIBuilder::createQualifiedType(unsigned Tag, DIType *FromTy) {
  return DIDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr, FromTy, 0,
                            0, 0, 0);
}

DIDerivedType *DIBuilder::createPointerType(DIType *PointeeTy,
                                            uint64_t SizeInBits,
                                            uint64_t AlignInBits,
                                            StringRef Name) {
  // FIXME: Why is there a name here?
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_pointer_type, Name,
                            nullptr, 0, nullptr, PointeeTy, SizeInBits,
                            AlignInBits, 0, 0);
}

DIDerivedType *DIBuilder::createMemberPointerType(DIType *PointeeTy,
                                                  DIType *Base,
                                                  uint64_t SizeInBits,
                                                  uint64_t AlignInBits,
                                                  unsigned Flags) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_ptr_to_member_type, "",
                            nullptr, 0, nullptr, PointeeTy, SizeInBits,
                            AlignInBits, 0, Flags, Base);
}

DIDerivedType *DIBuilder::createReferenceType(unsigned Tag, DIType *RTy,
                                              uint64_t SizeInBits,
                                              uint64_t AlignInBits) {
  assert(RTy && "Unable to create reference type");
  return DIDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr, RTy,
                            SizeInBits, AlignInBits, 0, 0);
}

DIDerivedType *DIBuilder::createTypedef(DIType *Ty, StringRef Name,
                                        DIFile *File, unsigned LineNo,
                                        DIScope *Context) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_typedef, Name, File,
                            LineNo, getNonCompileUnitScope(Context), Ty, 0, 0,
                            0, 0);
}

DIDerivedType *DIBuilder::createFriend(DIType *Ty, DIType *FriendTy) {
  assert(Ty && "Invalid type!");
  assert(FriendTy && "Invalid friend type!");
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_friend, "", nullptr, 0, Ty,
                            FriendTy, 0, 0, 0, 0);
}

DIDerivedType *DIBuilder::createInheritance(DIType *Ty, DIType *BaseTy,
                                            uint64_t BaseOffset,
                                            unsigned Flags) {
  assert(Ty && "Unable to create inheritance");
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_inheritance, "", nullptr,
                            0, Ty, BaseTy, 0, 0, BaseOffset, Flags);
}

DIDerivedType *DIBuilder::createMemberType(DIScope *Scope, StringRef Name,
                                           DIFile *File, unsigned LineNumber,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits,
                                           uint64_t OffsetInBits,
                                           unsigned Flags, DIType *Ty) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_member, Name, File,
                            LineNumber, getNonCompileUnitScope(Scope), Ty,
                            SizeInBits, AlignInBits, OffsetInBits, Flags);
}

static ConstantAsMetadata *getConstantOrNull(Constant *C) {
  if (C)
    return ConstantAsMetadata::get(C);
  return nullptr;
}

DIDerivedType *DIBuilder::createStaticMemberType(DIScope *Scope, StringRef Name,
                                                 DIFile *File,
                                                 unsigned LineNumber,
                                                 DIType *Ty, unsigned Flags,
                                                 llvm::Constant *Val) {
  Flags |= DINode::FlagStaticMember;
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_member, Name, File,
                            LineNumber, getNonCompileUnitScope(Scope), Ty, 0, 0,
                            0, Flags, getConstantOrNull(Val));
}

DIDerivedType *DIBuilder::createObjCIVar(StringRef Name, DIFile *File,
                                         unsigned LineNumber,
                                         uint64_t SizeInBits,
                                         uint64_t AlignInBits,
                                         uint64_t OffsetInBits, unsigned Flags,
                                         DIType *Ty, MDNode *PropertyNode) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_member, Name, File,
                            LineNumber, getNonCompileUnitScope(File), Ty,
                            SizeInBits, AlignInBits, OffsetInBits, Flags,
                            PropertyNode);
}

DIObjCProperty *
DIBuilder::createObjCProperty(StringRef Name, DIFile *File, unsigned LineNumber,
                              StringRef GetterName, StringRef SetterName,
                              unsigned PropertyAttributes, DIType *Ty) {
  return DIObjCProperty::get(VMContext, Name, File, LineNumber, GetterName,
                             SetterName, PropertyAttributes, Ty);
}

DITemplateTypeParameter *
DIBuilder::createTemplateTypeParameter(DIScope *Context, StringRef Name,
                                       DIType *Ty) {
  assert((!Context || isa<DICompileUnit>(Context)) && "Expected compile unit");
  return DITemplateTypeParameter::get(VMContext, Name, Ty);
}

static DITemplateValueParameter *
createTemplateValueParameterHelper(LLVMContext &VMContext, unsigned Tag,
                                   DIScope *Context, StringRef Name, DIType *Ty,
                                   Metadata *MD) {
  assert((!Context || isa<DICompileUnit>(Context)) && "Expected compile unit");
  return DITemplateValueParameter::get(VMContext, Tag, Name, Ty, MD);
}

DITemplateValueParameter *
DIBuilder::createTemplateValueParameter(DIScope *Context, StringRef Name,
                                        DIType *Ty, Constant *Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_template_value_parameter, Context, Name, Ty,
      getConstantOrNull(Val));
}

DITemplateValueParameter *
DIBuilder::createTemplateTemplateParameter(DIScope *Context, StringRef Name,
                                           DIType *Ty, StringRef Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_template_param, Context, Name, Ty,
      MDString::get(VMContext, Val));
}

DITemplateValueParameter *
DIBuilder::createTemplateParameterPack(DIScope *Context, StringRef Name,
                                       DIType *Ty, DINodeArray Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_parameter_pack, Context, Name, Ty,
      Val.get());
}

DICompositeType *DIBuilder::createClassType(
    DIScope *Context, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
    unsigned Flags, DIType *DerivedFrom, DINodeArray Elements,
    DIType *VTableHolder, MDNode *TemplateParams, StringRef UniqueIdentifier) {
  assert((!Context || isa<DIScope>(Context)) &&
         "createClassType should be called with a valid Context");

  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      getNonCompileUnitScope(Context), DerivedFrom, SizeInBits, AlignInBits,
      OffsetInBits, Flags, Elements, 0, VTableHolder,
      cast_or_null<MDTuple>(TemplateParams), UniqueIdentifier);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createStructType(
    DIScope *Context, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
    DIType *DerivedFrom, DINodeArray Elements, unsigned RunTimeLang,
    DIType *VTableHolder, StringRef UniqueIdentifier) {
  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      getNonCompileUnitScope(Context), DerivedFrom, SizeInBits, AlignInBits, 0,
      Flags, Elements, RunTimeLang, VTableHolder, nullptr, UniqueIdentifier);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createUnionType(
    DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
    DINodeArray Elements, unsigned RunTimeLang, StringRef UniqueIdentifier) {
  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_union_type, Name, File, LineNumber,
      getNonCompileUnitScope(Scope), nullptr, SizeInBits, AlignInBits, 0, Flags,
      Elements, RunTimeLang, nullptr, nullptr, UniqueIdentifier);
  trackIfUnresolved(R);
  return R;
}

DISubroutineType *DIBuilder::createSubroutineType(DITypeRefArray ParameterTypes,
                                                  unsigned Flags, unsigned CC) {
  return DISubroutineType::get(VMContext, Flags, CC, ParameterTypes);
}

DICompositeType *DIBuilder::createExternalTypeRef(unsigned Tag, DIFile *File,
                                                  StringRef UniqueIdentifier) {
  assert(!UniqueIdentifier.empty() && "external type ref without uid");
  return DICompositeType::get(VMContext, Tag, "", nullptr, 0, nullptr, nullptr,
                              0, 0, 0, DINode::FlagExternalTypeRef, nullptr, 0,
                              nullptr, nullptr, UniqueIdentifier);
}

DICompositeType *DIBuilder::createEnumerationType(
    DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, DINodeArray Elements,
    DIType *UnderlyingType, StringRef UniqueIdentifier) {
  auto *CTy = DICompositeType::get(
      VMContext, dwarf::DW_TAG_enumeration_type, Name, File, LineNumber,
      getNonCompileUnitScope(Scope), UnderlyingType, SizeInBits, AlignInBits, 0,
      0, Elements, 0, nullptr, nullptr, UniqueIdentifier);
  AllEnumTypes.push_back(CTy);
  trackIfUnresolved(CTy);
  return CTy;
}

DICompositeType *DIBuilder::createArrayType(uint64_t Size, uint64_t AlignInBits,
                                            DIType *Ty,
                                            DINodeArray Subscripts) {
  auto *R = DICompositeType::get(VMContext, dwarf::DW_TAG_array_type, "",
                                 nullptr, 0, nullptr, Ty, Size, AlignInBits, 0,
                                 0, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createVectorType(uint64_t Size,
                                             uint64_t AlignInBits, DIType *Ty,
                                             DINodeArray Subscripts) {
  auto *R = DICompositeType::get(VMContext, dwarf::DW_TAG_array_type, "",
                                 nullptr, 0, nullptr, Ty, Size, AlignInBits, 0,
                                 DINode::FlagVector, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

static DIType *createTypeWithFlags(LLVMContext &Context, DIType *Ty,
                                   unsigned FlagsToSet) {
  auto NewTy = Ty->clone();
  NewTy->setFlags(NewTy->getFlags() | FlagsToSet);
  return MDNode::replaceWithUniqued(std::move(NewTy));
}

DIType *DIBuilder::createArtificialType(DIType *Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty->isArtificial())
    return Ty;
  return createTypeWithFlags(VMContext, Ty, DINode::FlagArtificial);
}

DIType *DIBuilder::createObjectPointerType(DIType *Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty->isObjectPointer())
    return Ty;
  unsigned Flags = DINode::FlagObjectPointer | DINode::FlagArtificial;
  return createTypeWithFlags(VMContext, Ty, Flags);
}

void DIBuilder::retainType(DIScope *T) {
  assert(T && "Expected non-null type");
  assert((isa<DIType>(T) || (isa<DISubprogram>(T) &&
                             cast<DISubprogram>(T)->isDefinition() == false)) &&
         "Expected type or subprogram declaration");
  AllRetainTypes.emplace_back(T);
}

DIBasicType *DIBuilder::createUnspecifiedParameter() { return nullptr; }

DICompositeType *
DIBuilder::createForwardDecl(unsigned Tag, StringRef Name, DIScope *Scope,
                             DIFile *F, unsigned Line, unsigned RuntimeLang,
                             uint64_t SizeInBits, uint64_t AlignInBits,
                             StringRef UniqueIdentifier) {
  // FIXME: Define in terms of createReplaceableForwardDecl() by calling
  // replaceWithUniqued().
  auto *RetTy = DICompositeType::get(
      VMContext, Tag, Name, F, Line, getNonCompileUnitScope(Scope), nullptr,
      SizeInBits, AlignInBits, 0, DINode::FlagFwdDecl, nullptr, RuntimeLang,
      nullptr, nullptr, UniqueIdentifier);
  trackIfUnresolved(RetTy);
  return RetTy;
}

DICompositeType *DIBuilder::createReplaceableCompositeType(
    unsigned Tag, StringRef Name, DIScope *Scope, DIFile *F, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint64_t AlignInBits,
    unsigned Flags, StringRef UniqueIdentifier) {
  auto *RetTy =
      DICompositeType::getTemporary(
          VMContext, Tag, Name, F, Line, getNonCompileUnitScope(Scope), nullptr,
          SizeInBits, AlignInBits, 0, Flags, nullptr, RuntimeLang, nullptr,
          nullptr, UniqueIdentifier)
          .release();
  trackIfUnresolved(RetTy);
  return RetTy;
}

DINodeArray DIBuilder::getOrCreateArray(ArrayRef<Metadata *> Elements) {
  return MDTuple::get(VMContext, Elements);
}

DITypeRefArray DIBuilder::getOrCreateTypeArray(ArrayRef<Metadata *> Elements) {
  SmallVector<llvm::Metadata *, 16> Elts;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] && isa<MDNode>(Elements[i]))
      Elts.push_back(cast<DIType>(Elements[i]));
    else
      Elts.push_back(Elements[i]);
  }
  return DITypeRefArray(MDNode::get(VMContext, Elts));
}

DISubrange *DIBuilder::getOrCreateSubrange(int64_t Lo, int64_t Count) {
  return DISubrange::get(VMContext, Count, Lo);
}

static void checkGlobalVariableScope(DIScope *Context) {
#ifndef NDEBUG
  if (auto *CT =
          dyn_cast_or_null<DICompositeType>(getNonCompileUnitScope(Context)))
    assert(CT->getIdentifier().empty() &&
           "Context of a global variable should not be a type with identifier");
#endif
}

DIGlobalVariable *DIBuilder::createGlobalVariable(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *F,
    unsigned LineNumber, DIType *Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  auto *N = DIGlobalVariable::getDistinct(
      VMContext, cast_or_null<DIScope>(Context), Name, LinkageName, F,
      LineNumber, Ty, isLocalToUnit, true, Val,
      cast_or_null<DIDerivedType>(Decl));
  AllGVs.push_back(N);
  return N;
}

DIGlobalVariable *DIBuilder::createTempGlobalVariableFwdDecl(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *F,
    unsigned LineNumber, DIType *Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  return DIGlobalVariable::getTemporary(
             VMContext, cast_or_null<DIScope>(Context), Name, LinkageName, F,
             LineNumber, Ty, isLocalToUnit, false, Val,
             cast_or_null<DIDerivedType>(Decl))
      .release();
}

static DILocalVariable *createLocalVariable(
    LLVMContext &VMContext,
    DenseMap<MDNode *, SmallVector<TrackingMDNodeRef, 1>> &PreservedVariables,
    DIScope *Scope, StringRef Name, unsigned ArgNo, DIFile *File,
    unsigned LineNo, DIType *Ty, bool AlwaysPreserve, unsigned Flags) {
  // FIXME: Why getNonCompileUnitScope()?
  // FIXME: Why is "!Context" okay here?
  // FIXME: Why doesn't this check for a subprogram or lexical block (AFAICT
  // the only valid scopes)?
  DIScope *Context = getNonCompileUnitScope(Scope);

  auto *Node =
      DILocalVariable::get(VMContext, cast_or_null<DILocalScope>(Context), Name,
                           File, LineNo, Ty, ArgNo, Flags);
  if (AlwaysPreserve) {
    // The optimizer may remove local variables. If there is an interest
    // to preserve variable info in such situation then stash it in a
    // named mdnode.
    DISubprogram *Fn = getDISubprogram(Scope);
    assert(Fn && "Missing subprogram for local variable");
    PreservedVariables[Fn].emplace_back(Node);
  }
  return Node;
}

DILocalVariable *DIBuilder::createAutoVariable(DIScope *Scope, StringRef Name,
                                               DIFile *File, unsigned LineNo,
                                               DIType *Ty, bool AlwaysPreserve,
                                               unsigned Flags) {
  return createLocalVariable(VMContext, PreservedVariables, Scope, Name,
                             /* ArgNo */ 0, File, LineNo, Ty, AlwaysPreserve,
                             Flags);
}

DILocalVariable *DIBuilder::createParameterVariable(
    DIScope *Scope, StringRef Name, unsigned ArgNo, DIFile *File,
    unsigned LineNo, DIType *Ty, bool AlwaysPreserve, unsigned Flags) {
  assert(ArgNo && "Expected non-zero argument number for parameter");
  return createLocalVariable(VMContext, PreservedVariables, Scope, Name, ArgNo,
                             File, LineNo, Ty, AlwaysPreserve, Flags);
}

DIExpression *DIBuilder::createExpression(ArrayRef<uint64_t> Addr) {
  return DIExpression::get(VMContext, Addr);
}

DIExpression *DIBuilder::createExpression(ArrayRef<int64_t> Signed) {
  // TODO: Remove the callers of this signed version and delete.
  SmallVector<uint64_t, 8> Addr(Signed.begin(), Signed.end());
  return createExpression(Addr);
}

DIExpression *DIBuilder::createBitPieceExpression(unsigned OffsetInBytes,
                                                  unsigned SizeInBytes) {
  uint64_t Addr[] = {dwarf::DW_OP_bit_piece, OffsetInBytes, SizeInBytes};
  return DIExpression::get(VMContext, Addr);
}

template <class... Ts>
static DISubprogram *getSubprogram(bool IsDistinct, Ts &&... Args) {
  if (IsDistinct)
    return DISubprogram::getDistinct(std::forward<Ts>(Args)...);
  return DISubprogram::get(std::forward<Ts>(Args)...);
}

DISubprogram *DIBuilder::createFunction(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *File,
    unsigned LineNo, DISubroutineType *Ty, bool isLocalToUnit,
    bool isDefinition, unsigned ScopeLine, unsigned Flags, bool isOptimized,
    DITemplateParameterArray TParams, DISubprogram *Decl) {
  auto *Node = getSubprogram(
      /* IsDistinct = */ isDefinition, VMContext,
      getNonCompileUnitScope(Context), Name, LinkageName, File, LineNo, Ty,
      isLocalToUnit, isDefinition, ScopeLine, nullptr, 0, 0, Flags, isOptimized,
      isDefinition ? CUNode : nullptr, TParams, Decl,
      MDTuple::getTemporary(VMContext, None).release());

  if (isDefinition)
    AllSubprograms.push_back(Node);
  trackIfUnresolved(Node);
  return Node;
}

DISubprogram *DIBuilder::createTempFunctionFwdDecl(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *File,
    unsigned LineNo, DISubroutineType *Ty, bool isLocalToUnit,
    bool isDefinition, unsigned ScopeLine, unsigned Flags, bool isOptimized,
    DITemplateParameterArray TParams, DISubprogram *Decl) {
  return DISubprogram::getTemporary(
             VMContext, getNonCompileUnitScope(Context), Name, LinkageName,
             File, LineNo, Ty, isLocalToUnit, isDefinition, ScopeLine, nullptr,
             0, 0, Flags, isOptimized, isDefinition ? CUNode : nullptr, TParams,
             Decl, nullptr)
      .release();
}

DISubprogram *
DIBuilder::createMethod(DIScope *Context, StringRef Name, StringRef LinkageName,
                        DIFile *F, unsigned LineNo, DISubroutineType *Ty,
                        bool isLocalToUnit, bool isDefinition, unsigned VK,
                        unsigned VIndex, DIType *VTableHolder, unsigned Flags,
                        bool isOptimized, DITemplateParameterArray TParams) {
  assert(getNonCompileUnitScope(Context) &&
         "Methods should have both a Context and a context that isn't "
         "the compile unit.");
  // FIXME: Do we want to use different scope/lines?
  auto *SP = getSubprogram(
      /* IsDistinct = */ isDefinition, VMContext, cast<DIScope>(Context), Name,
      LinkageName, F, LineNo, Ty, isLocalToUnit, isDefinition, LineNo,
      VTableHolder, VK, VIndex, Flags, isOptimized,
      isDefinition ? CUNode : nullptr, TParams, nullptr, nullptr);

  if (isDefinition)
    AllSubprograms.push_back(SP);
  trackIfUnresolved(SP);
  return SP;
}

DINamespace *DIBuilder::createNameSpace(DIScope *Scope, StringRef Name,
                                        DIFile *File, unsigned LineNo) {
  return DINamespace::get(VMContext, getNonCompileUnitScope(Scope), File, Name,
                          LineNo);
}

DIModule *DIBuilder::createModule(DIScope *Scope, StringRef Name,
                                  StringRef ConfigurationMacros,
                                  StringRef IncludePath,
                                  StringRef ISysRoot) {
 return DIModule::get(VMContext, getNonCompileUnitScope(Scope), Name,
                      ConfigurationMacros, IncludePath, ISysRoot);
}

DILexicalBlockFile *DIBuilder::createLexicalBlockFile(DIScope *Scope,
                                                      DIFile *File,
                                                      unsigned Discriminator) {
  return DILexicalBlockFile::get(VMContext, Scope, File, Discriminator);
}

DILexicalBlock *DIBuilder::createLexicalBlock(DIScope *Scope, DIFile *File,
                                              unsigned Line, unsigned Col) {
  // Make these distinct, to avoid merging two lexical blocks on the same
  // file/line/column.
  return DILexicalBlock::getDistinct(VMContext, getNonCompileUnitScope(Scope),
                                     File, Line, Col);
}

static Value *getDbgIntrinsicValueImpl(LLVMContext &VMContext, Value *V) {
  assert(V && "no value passed to dbg intrinsic");
  return MetadataAsValue::get(VMContext, ValueAsMetadata::get(V));
}

static Instruction *withDebugLoc(Instruction *I, const DILocation *DL) {
  I->setDebugLoc(const_cast<DILocation *>(DL));
  return I;
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DILocalVariable *VarInfo,
                                      DIExpression *Expr, const DILocation *DL,
                                      Instruction *InsertBefore) {
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.declare");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return withDebugLoc(CallInst::Create(DeclareFn, Args, "", InsertBefore), DL);
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DILocalVariable *VarInfo,
                                      DIExpression *Expr, const DILocation *DL,
                                      BasicBlock *InsertAtEnd) {
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.declare");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};

  // If this block already has a terminator then insert this intrinsic
  // before the terminator.
  if (TerminatorInst *T = InsertAtEnd->getTerminator())
    return withDebugLoc(CallInst::Create(DeclareFn, Args, "", T), DL);
  return withDebugLoc(CallInst::Create(DeclareFn, Args, "", InsertAtEnd), DL);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DILocalVariable *VarInfo,
                                                DIExpression *Expr,
                                                const DILocation *DL,
                                                Instruction *InsertBefore) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.value");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return withDebugLoc(CallInst::Create(ValueFn, Args, "", InsertBefore), DL);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DILocalVariable *VarInfo,
                                                DIExpression *Expr,
                                                const DILocation *DL,
                                                BasicBlock *InsertAtEnd) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.value");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};

  return withDebugLoc(CallInst::Create(ValueFn, Args, "", InsertAtEnd), DL);
}

void DIBuilder::replaceVTableHolder(DICompositeType *&T,
                                    DICompositeType *VTableHolder) {
  {
    TypedTrackingMDRef<DICompositeType> N(T);
    N->replaceVTableHolder(VTableHolder);
    T = N.get();
  }

  // If this didn't create a self-reference, just return.
  if (T != VTableHolder)
    return;

  // Look for unresolved operands.  T will drop RAUW support, orphaning any
  // cycles underneath it.
  if (T->isResolved())
    for (const MDOperand &O : T->operands())
      if (auto *N = dyn_cast_or_null<MDNode>(O))
        trackIfUnresolved(N);
}

void DIBuilder::replaceArrays(DICompositeType *&T, DINodeArray Elements,
                              DINodeArray TParams) {
  {
    TypedTrackingMDRef<DICompositeType> N(T);
    if (Elements)
      N->replaceElements(Elements);
    if (TParams)
      N->replaceTemplateParams(DITemplateParameterArray(TParams));
    T = N.get();
  }

  // If T isn't resolved, there's no problem.
  if (!T->isResolved())
    return;

  // If T is resolved, it may be due to a self-reference cycle.  Track the
  // arrays explicitly if they're unresolved, or else the cycles will be
  // orphaned.
  if (Elements)
    trackIfUnresolved(Elements.get());
  if (TParams)
    trackIfUnresolved(TParams.get());
}
