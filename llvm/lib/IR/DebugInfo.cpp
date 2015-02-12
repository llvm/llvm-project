//===--- DebugInfo.cpp - Debug Information Helper Classes -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::dwarf;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

bool DIDescriptor::Verify() const {
  return DbgNode &&
         (DIDerivedType(DbgNode).Verify() ||
          DICompositeType(DbgNode).Verify() || DIBasicType(DbgNode).Verify() ||
          DIVariable(DbgNode).Verify() || DISubprogram(DbgNode).Verify() ||
          DIGlobalVariable(DbgNode).Verify() || DIFile(DbgNode).Verify() ||
          DICompileUnit(DbgNode).Verify() || DINameSpace(DbgNode).Verify() ||
          DILexicalBlock(DbgNode).Verify() ||
          DILexicalBlockFile(DbgNode).Verify() ||
          DISubrange(DbgNode).Verify() || DIEnumerator(DbgNode).Verify() ||
          DIObjCProperty(DbgNode).Verify() ||
          DITemplateTypeParameter(DbgNode).Verify() ||
          DITemplateValueParameter(DbgNode).Verify() ||
          DIImportedEntity(DbgNode).Verify() || DIExpression(DbgNode).Verify());
}

static Metadata *getField(const MDNode *DbgNode, unsigned Elt) {
  if (!DbgNode || Elt >= DbgNode->getNumOperands())
    return nullptr;
  return DbgNode->getOperand(Elt);
}

static MDNode *getNodeField(const MDNode *DbgNode, unsigned Elt) {
  return dyn_cast_or_null<MDNode>(getField(DbgNode, Elt));
}

static StringRef getStringField(const MDNode *DbgNode, unsigned Elt) {
  if (MDString *MDS = dyn_cast_or_null<MDString>(getField(DbgNode, Elt)))
    return MDS->getString();
  return StringRef();
}

StringRef DIDescriptor::getStringField(unsigned Elt) const {
  return ::getStringField(DbgNode, Elt);
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (auto *C = getConstantField(Elt))
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return CI->getZExtValue();

  return 0;
}

int64_t DIDescriptor::getInt64Field(unsigned Elt) const {
  if (auto *C = getConstantField(Elt))
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return CI->getZExtValue();

  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  MDNode *Field = getNodeField(DbgNode, Elt);
  return DIDescriptor(Field);
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  return dyn_cast_or_null<GlobalVariable>(getConstantField(Elt));
}

Constant *DIDescriptor::getConstantField(unsigned Elt) const {
  if (!DbgNode)
    return nullptr;

  if (Elt < DbgNode->getNumOperands())
    if (auto *C =
            dyn_cast_or_null<ConstantAsMetadata>(DbgNode->getOperand(Elt)))
      return C->getValue();
  return nullptr;
}

Function *DIDescriptor::getFunctionField(unsigned Elt) const {
  return dyn_cast_or_null<Function>(getConstantField(Elt));
}

void DIDescriptor::replaceFunctionField(unsigned Elt, Function *F) {
  if (!DbgNode)
    return;

  if (Elt < DbgNode->getNumOperands()) {
    MDNode *Node = const_cast<MDNode *>(DbgNode);
    Node->replaceOperandWith(Elt, F ? ConstantAsMetadata::get(F) : nullptr);
  }
}

static unsigned DIVariableInlinedAtIndex = 4;
MDNode *DIVariable::getInlinedAt() const {
  return getNodeField(DbgNode, DIVariableInlinedAtIndex);
}

/// \brief Return the size reported by the variable's type.
unsigned DIVariable::getSizeInBits(const DITypeIdentifierMap &Map) {
  DIType Ty = getType().resolve(Map);
  // Follow derived types until we reach a type that
  // reports back a size.
  while (Ty.isDerivedType() && !Ty.getSizeInBits()) {
    DIDerivedType DT(&*Ty);
    Ty = DT.getTypeDerivedFrom().resolve(Map);
  }
  assert(Ty.getSizeInBits() && "type with size 0");
  return Ty.getSizeInBits();
}

uint64_t DIExpression::getElement(unsigned Idx) const {
  unsigned I = Idx + 1;
  assert(I < getNumHeaderFields() &&
         "non-existing complex address element requested");
  return getHeaderFieldAs<int64_t>(I);
}

bool DIExpression::isVariablePiece() const {
  return getNumElements() && getElement(0) == dwarf::DW_OP_piece;
}

uint64_t DIExpression::getPieceOffset() const {
  assert(isVariablePiece());
  return getElement(1);
}

uint64_t DIExpression::getPieceSize() const {
  assert(isVariablePiece());
  return getElement(2);
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

bool DIDescriptor::isSubroutineType() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subroutine_type;
}

bool DIDescriptor::isBasicType() const {
  if (!DbgNode)
    return false;
  switch (getTag()) {
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_unspecified_type:
    return true;
  default:
    return false;
  }
}

bool DIDescriptor::isDerivedType() const {
  if (!DbgNode)
    return false;
  switch (getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_rvalue_reference_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_member:
  case dwarf::DW_TAG_inheritance:
  case dwarf::DW_TAG_friend:
    return true;
  default:
    // CompositeTypes are currently modelled as DerivedTypes.
    return isCompositeType();
  }
}

bool DIDescriptor::isCompositeType() const {
  if (!DbgNode)
    return false;
  switch (getTag()) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_class_type:
    return true;
  default:
    return false;
  }
}

bool DIDescriptor::isVariable() const {
  if (!DbgNode)
    return false;
  switch (getTag()) {
  case dwarf::DW_TAG_auto_variable:
  case dwarf::DW_TAG_arg_variable:
    return true;
  default:
    return false;
  }
}

bool DIDescriptor::isType() const {
  return isBasicType() || isCompositeType() || isDerivedType();
}

bool DIDescriptor::isSubprogram() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subprogram;
}

bool DIDescriptor::isGlobalVariable() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_variable ||
                     getTag() == dwarf::DW_TAG_constant);
}

bool DIDescriptor::isScope() const {
  if (!DbgNode)
    return false;
  switch (getTag()) {
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_namespace:
  case dwarf::DW_TAG_file_type:
    return true;
  default:
    break;
  }
  return isType();
}

bool DIDescriptor::isTemplateTypeParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_type_parameter;
}

bool DIDescriptor::isTemplateValueParameter() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_template_value_parameter ||
                     getTag() == dwarf::DW_TAG_GNU_template_template_param ||
                     getTag() == dwarf::DW_TAG_GNU_template_parameter_pack);
}

bool DIDescriptor::isCompileUnit() const {
  return DbgNode && getTag() == dwarf::DW_TAG_compile_unit;
}

bool DIDescriptor::isFile() const {
  return DbgNode && getTag() == dwarf::DW_TAG_file_type;
}

bool DIDescriptor::isNameSpace() const {
  return DbgNode && getTag() == dwarf::DW_TAG_namespace;
}

bool DIDescriptor::isLexicalBlockFile() const {
  return DbgNode && getTag() == dwarf::DW_TAG_lexical_block &&
         DbgNode->getNumOperands() == 3 && getNumHeaderFields() == 2;
}

bool DIDescriptor::isLexicalBlock() const {
  // FIXME: There are always exactly 4 header fields in DILexicalBlock, but
  // something relies on this returning true for DILexicalBlockFile.
  return DbgNode && getTag() == dwarf::DW_TAG_lexical_block &&
         DbgNode->getNumOperands() == 3 &&
         (getNumHeaderFields() == 2 || getNumHeaderFields() == 4);
}

bool DIDescriptor::isSubrange() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subrange_type;
}

bool DIDescriptor::isEnumerator() const {
  return DbgNode && getTag() == dwarf::DW_TAG_enumerator;
}

bool DIDescriptor::isObjCProperty() const {
  return DbgNode && getTag() == dwarf::DW_TAG_APPLE_property;
}

bool DIDescriptor::isImportedEntity() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_imported_module ||
                     getTag() == dwarf::DW_TAG_imported_declaration);
}

bool DIDescriptor::isExpression() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_expression);
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

void DIDescriptor::replaceAllUsesWith(LLVMContext &VMContext, DIDescriptor D) {

  assert(DbgNode && "Trying to replace an unverified type!");

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  const MDNode *DN = D;
  if (DbgNode == DN) {
    SmallVector<Metadata *, 10> Ops(DbgNode->getNumOperands());
    for (size_t i = 0; i != Ops.size(); ++i)
      Ops[i] = DbgNode->getOperand(i);
    DN = MDNode::get(VMContext, Ops);
  }

  auto *Node = cast<MDNodeFwdDecl>(const_cast<MDNode *>(DbgNode));
  Node->replaceAllUsesWith(const_cast<MDNode *>(DN));
  MDNode::deleteTemporary(Node);
  DbgNode = DN;
}

void DIDescriptor::replaceAllUsesWith(MDNode *D) {
  assert(DbgNode && "Trying to replace an unverified type!");
  assert(DbgNode != D && "This replacement should always happen");
  auto *Node = cast<MDNodeFwdDecl>(const_cast<MDNode *>(DbgNode));
  Node->replaceAllUsesWith(D);
  MDNode::deleteTemporary(Node);
}

bool DICompileUnit::Verify() const {
  if (!isCompileUnit())
    return false;

  // Don't bother verifying the compilation directory or producer string
  // as those could be empty.
  if (getFilename().empty())
    return false;

  return DbgNode->getNumOperands() == 7 && getNumHeaderFields() == 8;
}

bool DIObjCProperty::Verify() const {
  if (!isObjCProperty())
    return false;

  // Don't worry about the rest of the strings for now.
  return DbgNode->getNumOperands() == 3 && getNumHeaderFields() == 6;
}

/// \brief Check if a field at position Elt of a MDNode is a MDNode.
///
/// We currently allow an empty string and an integer.
/// But we don't allow a non-empty string in a MDNode field.
static bool fieldIsMDNode(const MDNode *DbgNode, unsigned Elt) {
  // FIXME: This function should return true, if the field is null or the field
  // is indeed a MDNode: return !Fld || isa<MDNode>(Fld).
  Metadata *Fld = getField(DbgNode, Elt);
  if (Fld && isa<MDString>(Fld) && !cast<MDString>(Fld)->getString().empty())
    return false;
  return true;
}

/// \brief Check if a field at position Elt of a MDNode is a MDString.
static bool fieldIsMDString(const MDNode *DbgNode, unsigned Elt) {
  Metadata *Fld = getField(DbgNode, Elt);
  return !Fld || isa<MDString>(Fld);
}

/// \brief Check if a value can be a reference to a type.
static bool isTypeRef(const Metadata *MD) {
  if (!MD)
    return true;
  if (auto *S = dyn_cast<MDString>(MD))
    return !S->getString().empty();
  if (auto *N = dyn_cast<MDNode>(MD))
    return DIType(N).isType();
  return false;
}

/// \brief Check if referenced field might be a type.
static bool fieldIsTypeRef(const MDNode *DbgNode, unsigned Elt) {
  return isTypeRef(dyn_cast_or_null<Metadata>(getField(DbgNode, Elt)));
}

/// \brief Check if a value can be a ScopeRef.
static bool isScopeRef(const Metadata *MD) {
  if (!MD)
    return true;
  if (auto *S = dyn_cast<MDString>(MD))
    return !S->getString().empty();
  return isa<MDNode>(MD);
}

/// \brief Check if a field at position Elt of a MDNode can be a ScopeRef.
static bool fieldIsScopeRef(const MDNode *DbgNode, unsigned Elt) {
  return isScopeRef(dyn_cast_or_null<Metadata>(getField(DbgNode, Elt)));
}

bool DIType::Verify() const {
  if (!isType())
    return false;
  // Make sure Context @ field 2 is MDNode.
  if (!fieldIsScopeRef(DbgNode, 2))
    return false;

  // FIXME: Sink this into the various subclass verifies.
  uint16_t Tag = getTag();
  if (!isBasicType() && Tag != dwarf::DW_TAG_const_type &&
      Tag != dwarf::DW_TAG_volatile_type && Tag != dwarf::DW_TAG_pointer_type &&
      Tag != dwarf::DW_TAG_ptr_to_member_type &&
      Tag != dwarf::DW_TAG_reference_type &&
      Tag != dwarf::DW_TAG_rvalue_reference_type &&
      Tag != dwarf::DW_TAG_restrict_type && Tag != dwarf::DW_TAG_array_type &&
      Tag != dwarf::DW_TAG_enumeration_type &&
      Tag != dwarf::DW_TAG_subroutine_type &&
      Tag != dwarf::DW_TAG_inheritance && Tag != dwarf::DW_TAG_friend &&
      getFilename().empty())
    return false;

  // DIType is abstract, it should be a BasicType, a DerivedType or
  // a CompositeType.
  if (isBasicType())
    return DIBasicType(DbgNode).Verify();
  else if (isCompositeType())
    return DICompositeType(DbgNode).Verify();
  else if (isDerivedType())
    return DIDerivedType(DbgNode).Verify();
  else
    return false;
}

bool DIBasicType::Verify() const {
  return isBasicType() && DbgNode->getNumOperands() == 3 &&
         getNumHeaderFields() == 8;
}

bool DIDerivedType::Verify() const {
  // Make sure DerivedFrom @ field 3 is TypeRef.
  if (!fieldIsTypeRef(DbgNode, 3))
    return false;
  if (getTag() == dwarf::DW_TAG_ptr_to_member_type)
    // Make sure ClassType @ field 4 is a TypeRef.
    if (!fieldIsTypeRef(DbgNode, 4))
      return false;

  return isDerivedType() && DbgNode->getNumOperands() >= 4 &&
         DbgNode->getNumOperands() <= 8 && getNumHeaderFields() >= 7 &&
         getNumHeaderFields() <= 8;
}

bool DICompositeType::Verify() const {
  if (!isCompositeType())
    return false;

  // Make sure DerivedFrom @ field 3 and ContainingType @ field 5 are TypeRef.
  if (!fieldIsTypeRef(DbgNode, 3))
    return false;
  if (!fieldIsTypeRef(DbgNode, 5))
    return false;

  // Make sure the type identifier at field 7 is MDString, it can be null.
  if (!fieldIsMDString(DbgNode, 7))
    return false;

  // A subroutine type can't be both & and &&.
  if (isLValueReference() && isRValueReference())
    return false;

  return DbgNode->getNumOperands() == 8 && getNumHeaderFields() == 8;
}

bool DISubprogram::Verify() const {
  if (!isSubprogram())
    return false;

  // Make sure context @ field 2 is a ScopeRef and type @ field 3 is a MDNode.
  if (!fieldIsScopeRef(DbgNode, 2))
    return false;
  if (!fieldIsMDNode(DbgNode, 3))
    return false;
  // Containing type @ field 4.
  if (!fieldIsTypeRef(DbgNode, 4))
    return false;

  // A subprogram can't be both & and &&.
  if (isLValueReference() && isRValueReference())
    return false;

  // If a DISubprogram has an llvm::Function*, then scope chains from all
  // instructions within the function should lead to this DISubprogram.
  if (auto *F = getFunction()) {
    for (auto &BB : *F) {
      for (auto &I : BB) {
        DebugLoc DL = I.getDebugLoc();
        if (DL.isUnknown())
          continue;

        MDNode *Scope = nullptr;
        MDNode *IA = nullptr;
        // walk the inlined-at scopes
        while ((IA = DL.getInlinedAt()))
          DL = DebugLoc::getFromDILocation(IA);
        DL.getScopeAndInlinedAt(Scope, IA);
        if (!Scope)
          return false;
        assert(!IA);
        while (!DIDescriptor(Scope).isSubprogram()) {
          DILexicalBlockFile D(Scope);
          Scope = D.isLexicalBlockFile()
                      ? D.getScope()
                      : DebugLoc::getFromDILexicalBlock(Scope).getScope();
          assert(Scope && "lexical block file has no scope");
        }
        if (!DISubprogram(Scope).describes(F))
          return false;
      }
    }
  }
  return DbgNode->getNumOperands() == 9 && getNumHeaderFields() == 12;
}

bool DIGlobalVariable::Verify() const {
  if (!isGlobalVariable())
    return false;

  if (getDisplayName().empty())
    return false;
  // Make sure context @ field 1 is an MDNode.
  if (!fieldIsMDNode(DbgNode, 1))
    return false;
  // Make sure that type @ field 3 is a DITypeRef.
  if (!fieldIsTypeRef(DbgNode, 3))
    return false;
  // Make sure StaticDataMemberDeclaration @ field 5 is MDNode.
  if (!fieldIsMDNode(DbgNode, 5))
    return false;

  return DbgNode->getNumOperands() == 6 && getNumHeaderFields() == 7;
}

bool DIVariable::Verify() const {
  if (!isVariable())
    return false;

  // Make sure context @ field 1 is an MDNode.
  if (!fieldIsMDNode(DbgNode, 1))
    return false;
  // Make sure that type @ field 3 is a DITypeRef.
  if (!fieldIsTypeRef(DbgNode, 3))
    return false;

  // Check the number of header fields, which is common between complex and
  // simple variables.
  if (getNumHeaderFields() != 4)
    return false;

  // Variable without an inline location.
  if (DbgNode->getNumOperands() == 4)
    return true;

  // Variable with an inline location.
  return getInlinedAt() != nullptr && DbgNode->getNumOperands() == 5;
}

bool DIExpression::Verify() const {
  // Empty DIExpressions may be represented as a nullptr.
  if (!DbgNode)
    return true;

  return isExpression() && DbgNode->getNumOperands() == 1;
}

bool DILocation::Verify() const {
  return DbgNode && isa<MDLocation>(DbgNode);
}

bool DINameSpace::Verify() const {
  if (!isNameSpace())
    return false;
  return DbgNode->getNumOperands() == 3 && getNumHeaderFields() == 3;
}

MDNode *DIFile::getFileNode() const { return getNodeField(DbgNode, 1); }

bool DIFile::Verify() const {
  return isFile() && DbgNode->getNumOperands() == 2;
}

bool DIEnumerator::Verify() const {
  return isEnumerator() && DbgNode->getNumOperands() == 1 &&
         getNumHeaderFields() == 3;
}

bool DISubrange::Verify() const {
  return isSubrange() && DbgNode->getNumOperands() == 1 &&
         getNumHeaderFields() == 3;
}

bool DILexicalBlock::Verify() const {
  return isLexicalBlock() && DbgNode->getNumOperands() == 3 &&
         getNumHeaderFields() == 4;
}

bool DILexicalBlockFile::Verify() const {
  return isLexicalBlockFile() && DbgNode->getNumOperands() == 3 &&
         getNumHeaderFields() == 2;
}

bool DITemplateTypeParameter::Verify() const {
  return isTemplateTypeParameter() && DbgNode->getNumOperands() == 4 &&
         getNumHeaderFields() == 4;
}

bool DITemplateValueParameter::Verify() const {
  return isTemplateValueParameter() && DbgNode->getNumOperands() == 5 &&
         getNumHeaderFields() == 4;
}

bool DIImportedEntity::Verify() const {
  return isImportedEntity() && DbgNode->getNumOperands() == 3 &&
         getNumHeaderFields() == 3;
}

MDNode *DIDerivedType::getObjCProperty() const {
  return getNodeField(DbgNode, 4);
}

MDString *DICompositeType::getIdentifier() const {
  return cast_or_null<MDString>(getField(DbgNode, 7));
}

#ifndef NDEBUG
static void VerifySubsetOf(const MDNode *LHS, const MDNode *RHS) {
  for (unsigned i = 0; i != LHS->getNumOperands(); ++i) {
    // Skip the 'empty' list (that's a single i32 0, rather than truly empty).
    if (i == 0 && mdconst::hasa<ConstantInt>(LHS->getOperand(i)))
      continue;
    const MDNode *E = cast<MDNode>(LHS->getOperand(i));
    bool found = false;
    for (unsigned j = 0; !found && j != RHS->getNumOperands(); ++j)
      found = (E == cast<MDNode>(RHS->getOperand(j)));
    assert(found && "Losing a member during member list replacement");
  }
}
#endif

void DICompositeType::setArraysHelper(MDNode *Elements, MDNode *TParams) {
  TrackingMDNodeRef N(*this);
  if (Elements) {
#ifndef NDEBUG
    // Check that the new list of members contains all the old members as well.
    if (const MDNode *El = cast_or_null<MDNode>(N->getOperand(4)))
      VerifySubsetOf(El, Elements);
#endif
    N->replaceOperandWith(4, Elements);
  }
  if (TParams)
    N->replaceOperandWith(6, TParams);
  DbgNode = N;
}

DIScopeRef DIScope::getRef() const {
  if (!isCompositeType())
    return DIScopeRef(*this);
  DICompositeType DTy(DbgNode);
  if (!DTy.getIdentifier())
    return DIScopeRef(*this);
  return DIScopeRef(DTy.getIdentifier());
}

void DICompositeType::setContainingType(DICompositeType ContainingType) {
  TrackingMDNodeRef N(*this);
  N->replaceOperandWith(5, ContainingType.getRef());
  DbgNode = N;
}

bool DIVariable::isInlinedFnArgument(const Function *CurFn) {
  assert(CurFn && "Invalid function");
  if (!getContext().isSubprogram())
    return false;
  // This variable is not inlined function argument if its scope
  // does not describe current function.
  return !DISubprogram(getContext()).describes(CurFn);
}

bool DISubprogram::describes(const Function *F) {
  assert(F && "Invalid function");
  if (F == getFunction())
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  if (F->getName() == Name)
    return true;
  return false;
}

MDNode *DISubprogram::getVariablesNodes() const {
  return getNodeField(DbgNode, 8);
}

DIArray DISubprogram::getVariables() const {
  return DIArray(getNodeField(DbgNode, 8));
}

Metadata *DITemplateValueParameter::getValue() const {
  return DbgNode->getOperand(3);
}

DIScopeRef DIScope::getContext() const {

  if (isType())
    return DIType(DbgNode).getContext();

  if (isSubprogram())
    return DIScopeRef(DISubprogram(DbgNode).getContext());

  if (isLexicalBlock())
    return DIScopeRef(DILexicalBlock(DbgNode).getContext());

  if (isLexicalBlockFile())
    return DIScopeRef(DILexicalBlockFile(DbgNode).getContext());

  if (isNameSpace())
    return DIScopeRef(DINameSpace(DbgNode).getContext());

  assert((isFile() || isCompileUnit()) && "Unhandled type of scope.");
  return DIScopeRef(nullptr);
}

StringRef DIScope::getName() const {
  if (isType())
    return DIType(DbgNode).getName();
  if (isSubprogram())
    return DISubprogram(DbgNode).getName();
  if (isNameSpace())
    return DINameSpace(DbgNode).getName();
  assert((isLexicalBlock() || isLexicalBlockFile() || isFile() ||
          isCompileUnit()) &&
         "Unhandled type of scope.");
  return StringRef();
}

StringRef DIScope::getFilename() const {
  if (!DbgNode)
    return StringRef();
  return ::getStringField(getNodeField(DbgNode, 1), 0);
}

StringRef DIScope::getDirectory() const {
  if (!DbgNode)
    return StringRef();
  return ::getStringField(getNodeField(DbgNode, 1), 1);
}

DIArray DICompileUnit::getEnumTypes() const {
  if (!DbgNode || DbgNode->getNumOperands() < 7)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 2));
}

DIArray DICompileUnit::getRetainedTypes() const {
  if (!DbgNode || DbgNode->getNumOperands() < 7)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 3));
}

DIArray DICompileUnit::getSubprograms() const {
  if (!DbgNode || DbgNode->getNumOperands() < 7)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 4));
}

DIArray DICompileUnit::getGlobalVariables() const {
  if (!DbgNode || DbgNode->getNumOperands() < 7)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 5));
}

DIArray DICompileUnit::getImportedEntities() const {
  if (!DbgNode || DbgNode->getNumOperands() < 7)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 6));
}

void DICompileUnit::replaceSubprograms(DIArray Subprograms) {
  assert(Verify() && "Expected compile unit");
  if (Subprograms == getSubprograms())
    return;

  const_cast<MDNode *>(DbgNode)->replaceOperandWith(4, Subprograms);
}

void DICompileUnit::replaceGlobalVariables(DIArray GlobalVariables) {
  assert(Verify() && "Expected compile unit");
  if (GlobalVariables == getGlobalVariables())
    return;

  const_cast<MDNode *>(DbgNode)->replaceOperandWith(5, GlobalVariables);
}

DILocation DILocation::copyWithNewScope(LLVMContext &Ctx,
                                        DILexicalBlockFile NewScope) {
  assert(Verify());
  assert(NewScope && "Expected valid scope");

  const auto *Old = cast<MDLocation>(DbgNode);
  return DILocation(MDLocation::get(Ctx, Old->getLine(), Old->getColumn(),
                                    NewScope, Old->getInlinedAt()));
}

unsigned DILocation::computeNewDiscriminator(LLVMContext &Ctx) {
  std::pair<const char *, unsigned> Key(getFilename().data(), getLineNumber());
  return ++Ctx.pImpl->DiscriminatorTable[Key];
}

DIVariable llvm::createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                       LLVMContext &VMContext) {
  assert(DIVariable(DV).Verify() && "Expected a DIVariable");
  if (!InlinedScope)
    return cleanseInlinedVariable(DV, VMContext);

  // Insert inlined scope.
  SmallVector<Metadata *, 8> Elts;
  for (unsigned I = 0, E = DIVariableInlinedAtIndex; I != E; ++I)
    Elts.push_back(DV->getOperand(I));
  Elts.push_back(InlinedScope);

  DIVariable Inlined(MDNode::get(VMContext, Elts));
  assert(Inlined.Verify() && "Expected to create a DIVariable");
  return Inlined;
}

DIVariable llvm::cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext) {
  assert(DIVariable(DV).Verify() && "Expected a DIVariable");
  if (!DIVariable(DV).getInlinedAt())
    return DIVariable(DV);

  // Remove inlined scope.
  SmallVector<Metadata *, 8> Elts;
  for (unsigned I = 0, E = DIVariableInlinedAtIndex; I != E; ++I)
    Elts.push_back(DV->getOperand(I));

  DIVariable Cleansed(MDNode::get(VMContext, Elts));
  assert(Cleansed.Verify() && "Expected to create a DIVariable");
  return Cleansed;
}

DISubprogram llvm::getDISubprogram(const MDNode *Scope) {
  DIDescriptor D(Scope);
  if (D.isSubprogram())
    return DISubprogram(Scope);

  if (D.isLexicalBlockFile())
    return getDISubprogram(DILexicalBlockFile(Scope).getContext());

  if (D.isLexicalBlock())
    return getDISubprogram(DILexicalBlock(Scope).getContext());

  return DISubprogram();
}

DISubprogram llvm::getDISubprogram(const Function *F) {
  // We look for the first instr that has a debug annotation leading back to F.
  for (auto &BB : *F) {
    auto Inst = std::find_if(BB.begin(), BB.end(), [](const Instruction &Inst) {
      return !Inst.getDebugLoc().isUnknown();
    });
    if (Inst == BB.end())
      continue;
    DebugLoc DLoc = Inst->getDebugLoc();
    const MDNode *Scope = DLoc.getScopeNode();
    DISubprogram Subprogram = getDISubprogram(Scope);
    return Subprogram.describes(F) ? Subprogram : DISubprogram();
  }

  return DISubprogram();
}

DICompositeType llvm::getDICompositeType(DIType T) {
  if (T.isCompositeType())
    return DICompositeType(T);

  if (T.isDerivedType()) {
    // This function is currently used by dragonegg and dragonegg does
    // not generate identifier for types, so using an empty map to resolve
    // DerivedFrom should be fine.
    DITypeIdentifierMap EmptyMap;
    return getDICompositeType(
        DIDerivedType(T).getTypeDerivedFrom().resolve(EmptyMap));
  }

  return DICompositeType();
}

DITypeIdentifierMap
llvm::generateDITypeIdentifierMap(const NamedMDNode *CU_Nodes) {
  DITypeIdentifierMap Map;
  for (unsigned CUi = 0, CUe = CU_Nodes->getNumOperands(); CUi != CUe; ++CUi) {
    DICompileUnit CU(CU_Nodes->getOperand(CUi));
    DIArray Retain = CU.getRetainedTypes();
    for (unsigned Ti = 0, Te = Retain.getNumElements(); Ti != Te; ++Ti) {
      if (!Retain.getElement(Ti).isCompositeType())
        continue;
      DICompositeType Ty(Retain.getElement(Ti));
      if (MDString *TypeId = Ty.getIdentifier()) {
        // Definition has priority over declaration.
        // Try to insert (TypeId, Ty) to Map.
        std::pair<DITypeIdentifierMap::iterator, bool> P =
            Map.insert(std::make_pair(TypeId, Ty));
        // If TypeId already exists in Map and this is a definition, replace
        // whatever we had (declaration or definition) with the definition.
        if (!P.second && !Ty.isForwardDecl())
          P.first->second = Ty;
      }
    }
  }
  return Map;
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

void DebugInfoFinder::reset() {
  CUs.clear();
  SPs.clear();
  GVs.clear();
  TYs.clear();
  Scopes.clear();
  NodesSeen.clear();
  TypeIdentifierMap.clear();
  TypeMapInitialized = false;
}

void DebugInfoFinder::InitializeTypeMap(const Module &M) {
  if (!TypeMapInitialized)
    if (NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu")) {
      TypeIdentifierMap = generateDITypeIdentifierMap(CU_Nodes);
      TypeMapInitialized = true;
    }
}

void DebugInfoFinder::processModule(const Module &M) {
  InitializeTypeMap(M);
  if (NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
      DICompileUnit CU(CU_Nodes->getOperand(i));
      addCompileUnit(CU);
      DIArray GVs = CU.getGlobalVariables();
      for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i) {
        DIGlobalVariable DIG(GVs.getElement(i));
        if (addGlobalVariable(DIG)) {
          processScope(DIG.getContext());
          processType(DIG.getType().resolve(TypeIdentifierMap));
        }
      }
      DIArray SPs = CU.getSubprograms();
      for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i)
        processSubprogram(DISubprogram(SPs.getElement(i)));
      DIArray EnumTypes = CU.getEnumTypes();
      for (unsigned i = 0, e = EnumTypes.getNumElements(); i != e; ++i)
        processType(DIType(EnumTypes.getElement(i)));
      DIArray RetainedTypes = CU.getRetainedTypes();
      for (unsigned i = 0, e = RetainedTypes.getNumElements(); i != e; ++i)
        processType(DIType(RetainedTypes.getElement(i)));
      DIArray Imports = CU.getImportedEntities();
      for (unsigned i = 0, e = Imports.getNumElements(); i != e; ++i) {
        DIImportedEntity Import = DIImportedEntity(Imports.getElement(i));
        DIDescriptor Entity = Import.getEntity().resolve(TypeIdentifierMap);
        if (Entity.isType())
          processType(DIType(Entity));
        else if (Entity.isSubprogram())
          processSubprogram(DISubprogram(Entity));
        else if (Entity.isNameSpace())
          processScope(DINameSpace(Entity).getContext());
      }
    }
  }
}

void DebugInfoFinder::processLocation(const Module &M, DILocation Loc) {
  if (!Loc)
    return;
  InitializeTypeMap(M);
  processScope(Loc.getScope());
  processLocation(M, Loc.getOrigLocation());
}

void DebugInfoFinder::processType(DIType DT) {
  if (!addType(DT))
    return;
  processScope(DT.getContext().resolve(TypeIdentifierMap));
  if (DT.isCompositeType()) {
    DICompositeType DCT(DT);
    processType(DCT.getTypeDerivedFrom().resolve(TypeIdentifierMap));
    if (DT.isSubroutineType()) {
      DITypeArray DTA = DISubroutineType(DT).getTypeArray();
      for (unsigned i = 0, e = DTA.getNumElements(); i != e; ++i)
        processType(DTA.getElement(i).resolve(TypeIdentifierMap));
      return;
    }
    DIArray DA = DCT.getElements();
    for (unsigned i = 0, e = DA.getNumElements(); i != e; ++i) {
      DIDescriptor D = DA.getElement(i);
      if (D.isType())
        processType(DIType(D));
      else if (D.isSubprogram())
        processSubprogram(DISubprogram(D));
    }
  } else if (DT.isDerivedType()) {
    DIDerivedType DDT(DT);
    processType(DDT.getTypeDerivedFrom().resolve(TypeIdentifierMap));
  }
}

void DebugInfoFinder::processScope(DIScope Scope) {
  if (Scope.isType()) {
    DIType Ty(Scope);
    processType(Ty);
    return;
  }
  if (Scope.isCompileUnit()) {
    addCompileUnit(DICompileUnit(Scope));
    return;
  }
  if (Scope.isSubprogram()) {
    processSubprogram(DISubprogram(Scope));
    return;
  }
  if (!addScope(Scope))
    return;
  if (Scope.isLexicalBlock()) {
    DILexicalBlock LB(Scope);
    processScope(LB.getContext());
  } else if (Scope.isLexicalBlockFile()) {
    DILexicalBlockFile LBF = DILexicalBlockFile(Scope);
    processScope(LBF.getScope());
  } else if (Scope.isNameSpace()) {
    DINameSpace NS(Scope);
    processScope(NS.getContext());
  }
}

void DebugInfoFinder::processSubprogram(DISubprogram SP) {
  if (!addSubprogram(SP))
    return;
  processScope(SP.getContext().resolve(TypeIdentifierMap));
  processType(SP.getType());
  DIArray TParams = SP.getTemplateParams();
  for (unsigned I = 0, E = TParams.getNumElements(); I != E; ++I) {
    DIDescriptor Element = TParams.getElement(I);
    if (Element.isTemplateTypeParameter()) {
      DITemplateTypeParameter TType(Element);
      processScope(TType.getContext().resolve(TypeIdentifierMap));
      processType(TType.getType().resolve(TypeIdentifierMap));
    } else if (Element.isTemplateValueParameter()) {
      DITemplateValueParameter TVal(Element);
      processScope(TVal.getContext().resolve(TypeIdentifierMap));
      processType(TVal.getType().resolve(TypeIdentifierMap));
    }
  }
}

void DebugInfoFinder::processDeclare(const Module &M,
                                     const DbgDeclareInst *DDI) {
  MDNode *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType().resolve(TypeIdentifierMap));
}

void DebugInfoFinder::processValue(const Module &M, const DbgValueInst *DVI) {
  MDNode *N = dyn_cast<MDNode>(DVI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType().resolve(TypeIdentifierMap));
}

bool DebugInfoFinder::addType(DIType DT) {
  if (!DT)
    return false;

  if (!NodesSeen.insert(DT).second)
    return false;

  TYs.push_back(DT);
  return true;
}

bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (!CU)
    return false;
  if (!NodesSeen.insert(CU).second)
    return false;

  CUs.push_back(CU);
  return true;
}

bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (!DIG)
    return false;

  if (!NodesSeen.insert(DIG).second)
    return false;

  GVs.push_back(DIG);
  return true;
}

bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (!SP)
    return false;

  if (!NodesSeen.insert(SP).second)
    return false;

  SPs.push_back(SP);
  return true;
}

bool DebugInfoFinder::addScope(DIScope Scope) {
  if (!Scope)
    return false;
  // FIXME: Ocaml binding generates a scope with no content, we treat it
  // as null for now.
  if (Scope->getNumOperands() == 0)
    return false;
  if (!NodesSeen.insert(Scope).second)
    return false;
  Scopes.push_back(Scope);
  return true;
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//

void DIDescriptor::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void DIDescriptor::print(raw_ostream &OS) const {
  if (!DbgNode)
    return;

  if (const char *Tag = dwarf::TagString(getTag()))
    OS << "[ " << Tag << " ]";

  if (this->isSubrange()) {
    DISubrange(DbgNode).printInternal(OS);
  } else if (this->isCompileUnit()) {
    DICompileUnit(DbgNode).printInternal(OS);
  } else if (this->isFile()) {
    DIFile(DbgNode).printInternal(OS);
  } else if (this->isEnumerator()) {
    DIEnumerator(DbgNode).printInternal(OS);
  } else if (this->isBasicType()) {
    DIType(DbgNode).printInternal(OS);
  } else if (this->isDerivedType()) {
    DIDerivedType(DbgNode).printInternal(OS);
  } else if (this->isCompositeType()) {
    DICompositeType(DbgNode).printInternal(OS);
  } else if (this->isSubprogram()) {
    DISubprogram(DbgNode).printInternal(OS);
  } else if (this->isGlobalVariable()) {
    DIGlobalVariable(DbgNode).printInternal(OS);
  } else if (this->isVariable()) {
    DIVariable(DbgNode).printInternal(OS);
  } else if (this->isObjCProperty()) {
    DIObjCProperty(DbgNode).printInternal(OS);
  } else if (this->isNameSpace()) {
    DINameSpace(DbgNode).printInternal(OS);
  } else if (this->isScope()) {
    DIScope(DbgNode).printInternal(OS);
  } else if (this->isExpression()) {
    DIExpression(DbgNode).printInternal(OS);
  }
}

void DISubrange::printInternal(raw_ostream &OS) const {
  int64_t Count = getCount();
  if (Count != -1)
    OS << " [" << getLo() << ", " << Count - 1 << ']';
  else
    OS << " [unbounded]";
}

void DIScope::printInternal(raw_ostream &OS) const {
  OS << " [" << getDirectory() << "/" << getFilename() << ']';
}

void DICompileUnit::printInternal(raw_ostream &OS) const {
  DIScope::printInternal(OS);
  OS << " [";
  unsigned Lang = getLanguage();
  if (const char *LangStr = dwarf::LanguageString(Lang))
    OS << LangStr;
  else
    (OS << "lang 0x").write_hex(Lang);
  OS << ']';
}

void DIEnumerator::printInternal(raw_ostream &OS) const {
  OS << " [" << getName() << " :: " << getEnumValue() << ']';
}

void DIType::printInternal(raw_ostream &OS) const {
  if (!DbgNode)
    return;

  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "]";

  // TODO: Print context?

  OS << " [line " << getLineNumber() << ", size " << getSizeInBits()
     << ", align " << getAlignInBits() << ", offset " << getOffsetInBits();
  if (isBasicType())
    if (const char *Enc =
            dwarf::AttributeEncodingString(DIBasicType(DbgNode).getEncoding()))
      OS << ", enc " << Enc;
  OS << "]";

  if (isPrivate())
    OS << " [private]";
  else if (isProtected())
    OS << " [protected]";
  else if (isPublic())
    OS << " [public]";

  if (isArtificial())
    OS << " [artificial]";

  if (isForwardDecl())
    OS << " [decl]";
  else if (getTag() == dwarf::DW_TAG_structure_type ||
           getTag() == dwarf::DW_TAG_union_type ||
           getTag() == dwarf::DW_TAG_enumeration_type ||
           getTag() == dwarf::DW_TAG_class_type)
    OS << " [def]";
  if (isVector())
    OS << " [vector]";
  if (isStaticMember())
    OS << " [static]";

  if (isLValueReference())
    OS << " [reference]";

  if (isRValueReference())
    OS << " [rvalue reference]";
}

void DIDerivedType::printInternal(raw_ostream &OS) const {
  DIType::printInternal(OS);
  OS << " [from " << getTypeDerivedFrom().getName() << ']';
}

void DICompositeType::printInternal(raw_ostream &OS) const {
  DIType::printInternal(OS);
  DIArray A = getElements();
  OS << " [" << A.getNumElements() << " elements]";
}

void DINameSpace::printInternal(raw_ostream &OS) const {
  StringRef Name = getName();
  if (!Name.empty())
    OS << " [" << Name << ']';

  OS << " [line " << getLineNumber() << ']';
}

void DISubprogram::printInternal(raw_ostream &OS) const {
  // TODO : Print context
  OS << " [line " << getLineNumber() << ']';

  if (isLocalToUnit())
    OS << " [local]";

  if (isDefinition())
    OS << " [def]";

  if (getScopeLineNumber() != getLineNumber())
    OS << " [scope " << getScopeLineNumber() << "]";

  if (isPrivate())
    OS << " [private]";
  else if (isProtected())
    OS << " [protected]";
  else if (isPublic())
    OS << " [public]";

  if (isLValueReference())
    OS << " [reference]";

  if (isRValueReference())
    OS << " [rvalue reference]";

  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << ']';
}

void DIGlobalVariable::printInternal(raw_ostream &OS) const {
  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << ']';

  OS << " [line " << getLineNumber() << ']';

  // TODO : Print context

  if (isLocalToUnit())
    OS << " [local]";

  if (isDefinition())
    OS << " [def]";
}

void DIVariable::printInternal(raw_ostream &OS) const {
  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << ']';

  OS << " [line " << getLineNumber() << ']';
}

void DIExpression::printInternal(raw_ostream &OS) const {
  for (unsigned I = 0; I < getNumElements(); ++I) {
    uint64_t OpCode = getElement(I);
    OS << " [" << OperationEncodingString(OpCode);
    switch (OpCode) {
    case DW_OP_plus: {
      OS << " " << getElement(++I);
      break;
    }
    case DW_OP_piece: {
      unsigned Offset = getElement(++I);
      unsigned Size = getElement(++I);
      OS << " offset=" << Offset << ", size=" << Size;
      break;
    }
    case DW_OP_deref:
      // No arguments.
      break;
    default:
      // Else bail out early. This may be a line table entry.
      OS << "Unknown]";
      return;
    }
    OS << "]";
  }
}

void DIObjCProperty::printInternal(raw_ostream &OS) const {
  StringRef Name = getObjCPropertyName();
  if (!Name.empty())
    OS << " [" << Name << ']';

  OS << " [line " << getLineNumber() << ", properties " << getUnsignedField(6)
     << ']';
}

static void printDebugLoc(DebugLoc DL, raw_ostream &CommentOS,
                          const LLVMContext &Ctx) {
  if (!DL.isUnknown()) { // Print source line info.
    DIScope Scope(DL.getScope(Ctx));
    assert(Scope.isScope() && "Scope of a DebugLoc should be a DIScope.");
    // Omit the directory, because it's likely to be long and uninteresting.
    CommentOS << Scope.getFilename();
    CommentOS << ':' << DL.getLine();
    if (DL.getCol() != 0)
      CommentOS << ':' << DL.getCol();
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(DL.getInlinedAt(Ctx));
    if (!InlinedAtDL.isUnknown()) {
      CommentOS << " @[ ";
      printDebugLoc(InlinedAtDL, CommentOS, Ctx);
      CommentOS << " ]";
    }
  }
}

void DIVariable::printExtendedName(raw_ostream &OS) const {
  const LLVMContext &Ctx = DbgNode->getContext();
  StringRef Res = getName();
  if (!Res.empty())
    OS << Res << "," << getLineNumber();
  if (MDNode *InlinedAt = getInlinedAt()) {
    DebugLoc InlinedAtDL = DebugLoc::getFromDILocation(InlinedAt);
    if (!InlinedAtDL.isUnknown()) {
      OS << " @[";
      printDebugLoc(InlinedAtDL, OS, Ctx);
      OS << "]";
    }
  }
}

template <> DIRef<DIScope>::DIRef(const Metadata *V) : Val(V) {
  assert(isScopeRef(V) && "DIScopeRef should be a MDString or MDNode");
}
template <> DIRef<DIType>::DIRef(const Metadata *V) : Val(V) {
  assert(isTypeRef(V) && "DITypeRef should be a MDString or MDNode");
}

template <>
DIScopeRef DIDescriptor::getFieldAs<DIScopeRef>(unsigned Elt) const {
  return DIScopeRef(cast_or_null<Metadata>(getField(DbgNode, Elt)));
}
template <> DITypeRef DIDescriptor::getFieldAs<DITypeRef>(unsigned Elt) const {
  return DITypeRef(cast_or_null<Metadata>(getField(DbgNode, Elt)));
}

bool llvm::StripDebugInfo(Module &M) {
  bool Changed = false;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (Function *Declare = M.getFunction("llvm.dbg.declare")) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->user_back());
      CI->eraseFromParent();
    }
    Declare->eraseFromParent();
    Changed = true;
  }

  if (Function *DbgVal = M.getFunction("llvm.dbg.value")) {
    while (!DbgVal->use_empty()) {
      CallInst *CI = cast<CallInst>(DbgVal->user_back());
      CI->eraseFromParent();
    }
    DbgVal->eraseFromParent();
    Changed = true;
  }

  for (Module::named_metadata_iterator NMI = M.named_metadata_begin(),
         NME = M.named_metadata_end(); NMI != NME;) {
    NamedMDNode *NMD = NMI;
    ++NMI;
    if (NMD->getName().startswith("llvm.dbg.")) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  for (Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    for (Function::iterator FI = MI->begin(), FE = MI->end(); FI != FE;
         ++FI)
      for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE;
           ++BI) {
        if (!BI->getDebugLoc().isUnknown()) {
          Changed = true;
          BI->setDebugLoc(DebugLoc());
        }
      }

  return Changed;
}

unsigned llvm::getDebugMetadataVersionFromModule(const Module &M) {
  if (auto *Val = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("Debug Info Version")))
    return Val->getZExtValue();
  return 0;
}

llvm::DenseMap<const llvm::Function *, llvm::DISubprogram>
llvm::makeSubprogramMap(const Module &M) {
  DenseMap<const Function *, DISubprogram> R;

  NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return R;

  for (MDNode *N : CU_Nodes->operands()) {
    DICompileUnit CUNode(N);
    DIArray SPs = CUNode.getSubprograms();
    for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i) {
      DISubprogram SP(SPs.getElement(i));
      if (Function *F = SP.getFunction())
        R.insert(std::make_pair(F, SP));
    }
  }
  return R;
}
