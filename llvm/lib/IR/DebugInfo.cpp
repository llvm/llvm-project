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

#include "llvm/DebugInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ValueHandle.h"
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
          DIImportedEntity(DbgNode).Verify());
}

static Value *getField(const MDNode *DbgNode, unsigned Elt) {
  if (DbgNode == 0 || Elt >= DbgNode->getNumOperands())
    return 0;
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
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    if (ConstantInt *CI =
            dyn_cast_or_null<ConstantInt>(DbgNode->getOperand(Elt)))
      return CI->getZExtValue();

  return 0;
}

int64_t DIDescriptor::getInt64Field(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    if (ConstantInt *CI =
            dyn_cast_or_null<ConstantInt>(DbgNode->getOperand(Elt)))
      return CI->getSExtValue();

  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  MDNode *Field = getNodeField(DbgNode, Elt);
  return DIDescriptor(Field);
}

GlobalVariable *DIDescriptor::getGlobalVariableField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    return dyn_cast_or_null<GlobalVariable>(DbgNode->getOperand(Elt));
  return 0;
}

Constant *DIDescriptor::getConstantField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    return dyn_cast_or_null<Constant>(DbgNode->getOperand(Elt));
  return 0;
}

Function *DIDescriptor::getFunctionField(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    return dyn_cast_or_null<Function>(DbgNode->getOperand(Elt));
  return 0;
}

void DIDescriptor::replaceFunctionField(unsigned Elt, Function *F) {
  if (DbgNode == 0)
    return;

  if (Elt < DbgNode->getNumOperands()) {
    MDNode *Node = const_cast<MDNode *>(DbgNode);
    Node->replaceOperandWith(Elt, F);
  }
}

unsigned DIVariable::getNumAddrElements() const {
  return DbgNode->getNumOperands() - 8;
}

/// getInlinedAt - If this variable is inlined then return inline location.
MDNode *DIVariable::getInlinedAt() const { return getNodeField(DbgNode, 7); }

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// isBasicType - Return true if the specified tag is legal for
/// DIBasicType.
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

/// isDerivedType - Return true if the specified tag is legal for DIDerivedType.
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

/// isCompositeType - Return true if the specified tag is legal for
/// DICompositeType.
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

/// isVariable - Return true if the specified tag is legal for DIVariable.
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

/// isType - Return true if the specified tag is legal for DIType.
bool DIDescriptor::isType() const {
  return isBasicType() || isCompositeType() || isDerivedType();
}

/// isSubprogram - Return true if the specified tag is legal for
/// DISubprogram.
bool DIDescriptor::isSubprogram() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subprogram;
}

/// isGlobalVariable - Return true if the specified tag is legal for
/// DIGlobalVariable.
bool DIDescriptor::isGlobalVariable() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_variable ||
                     getTag() == dwarf::DW_TAG_constant);
}

/// isUnspecifiedParmeter - Return true if the specified tag is
/// DW_TAG_unspecified_parameters.
bool DIDescriptor::isUnspecifiedParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_unspecified_parameters;
}

/// isScope - Return true if the specified tag is one of the scope
/// related tag.
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

/// isTemplateTypeParameter - Return true if the specified tag is
/// DW_TAG_template_type_parameter.
bool DIDescriptor::isTemplateTypeParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_type_parameter;
}

/// isTemplateValueParameter - Return true if the specified tag is
/// DW_TAG_template_value_parameter.
bool DIDescriptor::isTemplateValueParameter() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_template_value_parameter ||
                     getTag() == dwarf::DW_TAG_GNU_template_template_param ||
                     getTag() == dwarf::DW_TAG_GNU_template_parameter_pack);
}

/// isCompileUnit - Return true if the specified tag is DW_TAG_compile_unit.
bool DIDescriptor::isCompileUnit() const {
  return DbgNode && getTag() == dwarf::DW_TAG_compile_unit;
}

/// isFile - Return true if the specified tag is DW_TAG_file_type.
bool DIDescriptor::isFile() const {
  return DbgNode && getTag() == dwarf::DW_TAG_file_type;
}

/// isNameSpace - Return true if the specified tag is DW_TAG_namespace.
bool DIDescriptor::isNameSpace() const {
  return DbgNode && getTag() == dwarf::DW_TAG_namespace;
}

/// isLexicalBlockFile - Return true if the specified descriptor is a
/// lexical block with an extra file.
bool DIDescriptor::isLexicalBlockFile() const {
  return DbgNode && getTag() == dwarf::DW_TAG_lexical_block &&
         (DbgNode->getNumOperands() == 3);
}

/// isLexicalBlock - Return true if the specified tag is DW_TAG_lexical_block.
bool DIDescriptor::isLexicalBlock() const {
  return DbgNode && getTag() == dwarf::DW_TAG_lexical_block &&
         (DbgNode->getNumOperands() > 3);
}

/// isSubrange - Return true if the specified tag is DW_TAG_subrange_type.
bool DIDescriptor::isSubrange() const {
  return DbgNode && getTag() == dwarf::DW_TAG_subrange_type;
}

/// isEnumerator - Return true if the specified tag is DW_TAG_enumerator.
bool DIDescriptor::isEnumerator() const {
  return DbgNode && getTag() == dwarf::DW_TAG_enumerator;
}

/// isObjCProperty - Return true if the specified tag is DW_TAG_APPLE_property.
bool DIDescriptor::isObjCProperty() const {
  return DbgNode && getTag() == dwarf::DW_TAG_APPLE_property;
}

/// \brief Return true if the specified tag is DW_TAG_imported_module or
/// DW_TAG_imported_declaration.
bool DIDescriptor::isImportedEntity() const {
  return DbgNode && (getTag() == dwarf::DW_TAG_imported_module ||
                     getTag() == dwarf::DW_TAG_imported_declaration);
}

//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

unsigned DIArray::getNumElements() const {
  if (!DbgNode)
    return 0;
  return DbgNode->getNumOperands();
}

/// replaceAllUsesWith - Replace all uses of the MDNode used by this
/// type with the one in the passed descriptor.
void DIType::replaceAllUsesWith(DIDescriptor &D) {

  assert(DbgNode && "Trying to replace an unverified type!");

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode *>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value *>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// replaceAllUsesWith - Replace all uses of the MDNode used by this
/// type with the one in D.
void DIType::replaceAllUsesWith(MDNode *D) {

  assert(DbgNode && "Trying to replace an unverified type!");

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode *>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value *>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// Verify - Verify that a compile unit is well formed.
bool DICompileUnit::Verify() const {
  if (!isCompileUnit())
    return false;

  // Don't bother verifying the compilation directory or producer string
  // as those could be empty.
  if (getFilename().empty())
    return false;

  return DbgNode->getNumOperands() == 13;
}

/// Verify - Verify that an ObjC property is well formed.
bool DIObjCProperty::Verify() const {
  if (!isObjCProperty())
    return false;

  // Don't worry about the rest of the strings for now.
  return DbgNode->getNumOperands() == 8;
}

/// Check if a field at position Elt of a MDNode is a MDNode.
/// We currently allow an empty string and an integer.
/// But we don't allow a non-empty string in a MDNode field.
static bool fieldIsMDNode(const MDNode *DbgNode, unsigned Elt) {
  // FIXME: This function should return true, if the field is null or the field
  // is indeed a MDNode: return !Fld || isa<MDNode>(Fld).
  Value *Fld = getField(DbgNode, Elt);
  if (Fld && isa<MDString>(Fld) && !cast<MDString>(Fld)->getString().empty())
    return false;
  return true;
}

/// Check if a field at position Elt of a MDNode is a MDString.
static bool fieldIsMDString(const MDNode *DbgNode, unsigned Elt) {
  Value *Fld = getField(DbgNode, Elt);
  return !Fld || isa<MDString>(Fld);
}

/// Check if a value can be a reference to a type.
static bool isTypeRef(const Value *Val) {
  return !Val ||
         (isa<MDString>(Val) && !cast<MDString>(Val)->getString().empty()) ||
         (isa<MDNode>(Val) && DIType(cast<MDNode>(Val)).isType());
}

/// Check if a field at position Elt of a MDNode can be a reference to a type.
static bool fieldIsTypeRef(const MDNode *DbgNode, unsigned Elt) {
  Value *Fld = getField(DbgNode, Elt);
  return isTypeRef(Fld);
}

/// Check if a value can be a ScopeRef.
static bool isScopeRef(const Value *Val) {
  return !Val ||
         (isa<MDString>(Val) && !cast<MDString>(Val)->getString().empty()) ||
         (isa<MDNode>(Val) && DIScope(cast<MDNode>(Val)).isScope());
}

/// Check if a field at position Elt of a MDNode can be a ScopeRef.
static bool fieldIsScopeRef(const MDNode *DbgNode, unsigned Elt) {
  Value *Fld = getField(DbgNode, Elt);
  return isScopeRef(Fld);
}

/// Verify - Verify that a type descriptor is well formed.
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
    DIBasicType(DbgNode).Verify();
  else if (isCompositeType())
    DICompositeType(DbgNode).Verify();
  else if (isDerivedType())
    DIDerivedType(DbgNode).Verify();
  else
    return false;
  return true;
}

/// Verify - Verify that a basic type descriptor is well formed.
bool DIBasicType::Verify() const {
  return isBasicType() && DbgNode->getNumOperands() == 10;
}

/// Verify - Verify that a derived type descriptor is well formed.
bool DIDerivedType::Verify() const {
  // Make sure DerivedFrom @ field 9 is TypeRef.
  if (!fieldIsTypeRef(DbgNode, 9))
    return false;
  if (getTag() == dwarf::DW_TAG_ptr_to_member_type)
    // Make sure ClassType @ field 10 is a TypeRef.
    if (!fieldIsTypeRef(DbgNode, 10))
      return false;

  return isDerivedType() && DbgNode->getNumOperands() >= 10 &&
         DbgNode->getNumOperands() <= 14;
}

/// Verify - Verify that a composite type descriptor is well formed.
bool DICompositeType::Verify() const {
  if (!isCompositeType())
    return false;

  // Make sure DerivedFrom @ field 9 and ContainingType @ field 12 are TypeRef.
  if (!fieldIsTypeRef(DbgNode, 9))
    return false;
  if (!fieldIsTypeRef(DbgNode, 12))
    return false;

  // Make sure the type identifier at field 14 is MDString, it can be null.
  if (!fieldIsMDString(DbgNode, 14))
    return false;

  return DbgNode->getNumOperands() == 15;
}

/// Verify - Verify that a subprogram descriptor is well formed.
bool DISubprogram::Verify() const {
  if (!isSubprogram())
    return false;

  // Make sure context @ field 2 is a ScopeRef and type @ field 7 is a MDNode.
  if (!fieldIsScopeRef(DbgNode, 2))
    return false;
  if (!fieldIsMDNode(DbgNode, 7))
    return false;
  // Containing type @ field 12.
  if (!fieldIsTypeRef(DbgNode, 12))
    return false;
  return DbgNode->getNumOperands() == 20;
}

/// Verify - Verify that a global variable descriptor is well formed.
bool DIGlobalVariable::Verify() const {
  if (!isGlobalVariable())
    return false;

  if (getDisplayName().empty())
    return false;
  // Make sure context @ field 2 and type @ field 8 are MDNodes.
  if (!fieldIsMDNode(DbgNode, 2))
    return false;
  if (!fieldIsMDNode(DbgNode, 8))
    return false;
  // Make sure StaticDataMemberDeclaration @ field 12 is MDNode.
  if (!fieldIsMDNode(DbgNode, 12))
    return false;

  return DbgNode->getNumOperands() == 13;
}

/// Verify - Verify that a variable descriptor is well formed.
bool DIVariable::Verify() const {
  if (!isVariable())
    return false;

  // Make sure context @ field 1 and type @ field 5 are MDNodes.
  if (!fieldIsMDNode(DbgNode, 1))
    return false;
  if (!fieldIsMDNode(DbgNode, 5))
    return false;
  return DbgNode->getNumOperands() >= 8;
}

/// Verify - Verify that a location descriptor is well formed.
bool DILocation::Verify() const {
  if (!DbgNode)
    return false;

  return DbgNode->getNumOperands() == 4;
}

/// Verify - Verify that a namespace descriptor is well formed.
bool DINameSpace::Verify() const {
  if (!isNameSpace())
    return false;
  return DbgNode->getNumOperands() == 5;
}

/// \brief Retrieve the MDNode for the directory/file pair.
MDNode *DIFile::getFileNode() const { return getNodeField(DbgNode, 1); }

/// \brief Verify that the file descriptor is well formed.
bool DIFile::Verify() const {
  return isFile() && DbgNode->getNumOperands() == 2;
}

/// \brief Verify that the enumerator descriptor is well formed.
bool DIEnumerator::Verify() const {
  return isEnumerator() && DbgNode->getNumOperands() == 3;
}

/// \brief Verify that the subrange descriptor is well formed.
bool DISubrange::Verify() const {
  return isSubrange() && DbgNode->getNumOperands() == 3;
}

/// \brief Verify that the lexical block descriptor is well formed.
bool DILexicalBlock::Verify() const {
  return isLexicalBlock() && DbgNode->getNumOperands() == 6;
}

/// \brief Verify that the file-scoped lexical block descriptor is well formed.
bool DILexicalBlockFile::Verify() const {
  return isLexicalBlockFile() && DbgNode->getNumOperands() == 3;
}

/// \brief Verify that the template type parameter descriptor is well formed.
bool DITemplateTypeParameter::Verify() const {
  return isTemplateTypeParameter() && DbgNode->getNumOperands() == 7;
}

/// \brief Verify that the template value parameter descriptor is well formed.
bool DITemplateValueParameter::Verify() const {
  return isTemplateValueParameter() && DbgNode->getNumOperands() == 8;
}

/// \brief Verify that the imported module descriptor is well formed.
bool DIImportedEntity::Verify() const {
  return isImportedEntity() &&
         (DbgNode->getNumOperands() == 4 || DbgNode->getNumOperands() == 5);
}

/// getObjCProperty - Return property node, if this ivar is associated with one.
MDNode *DIDerivedType::getObjCProperty() const {
  return getNodeField(DbgNode, 10);
}

MDString *DICompositeType::getIdentifier() const {
  return cast_or_null<MDString>(getField(DbgNode, 14));
}

#ifndef NDEBUG
static void VerifySubsetOf(const MDNode *LHS, const MDNode *RHS) {
  for (unsigned i = 0; i != LHS->getNumOperands(); ++i) {
    // Skip the 'empty' list (that's a single i32 0, rather than truly empty).
    if (i == 0 && isa<ConstantInt>(LHS->getOperand(i)))
      continue;
    const MDNode *E = cast<MDNode>(LHS->getOperand(i));
    bool found = false;
    for (unsigned j = 0; !found && j != RHS->getNumOperands(); ++j)
      found = E == RHS->getOperand(j);
    assert(found && "Losing a member during member list replacement");
  }
}
#endif

/// \brief Set the array of member DITypes.
void DICompositeType::setTypeArray(DIArray Elements, DIArray TParams) {
  assert((!TParams || DbgNode->getNumOperands() == 15) &&
         "If you're setting the template parameters this should include a slot "
         "for that!");
  TrackingVH<MDNode> N(*this);
  if (Elements) {
#ifndef NDEBUG
    // Check that the new list of members contains all the old members as well.
    if (const MDNode *El = cast_or_null<MDNode>(N->getOperand(10)))
      VerifySubsetOf(El, Elements);
#endif
    N->replaceOperandWith(10, Elements);
  }
  if (TParams)
    N->replaceOperandWith(13, TParams);
  DbgNode = N;
}

void DICompositeType::addMember(DIDescriptor D) {
  SmallVector<llvm::Value *, 16> M;
  DIArray OrigM = getTypeArray();
  unsigned Elements = OrigM.getNumElements();
  if (Elements == 1 && !OrigM.getElement(0))
    Elements = 0;
  M.reserve(Elements + 1);
  for (unsigned i = 0; i != Elements; ++i)
    M.push_back(OrigM.getElement(i));
  M.push_back(D);
  setTypeArray(DIArray(MDNode::get(DbgNode->getContext(), M)));
}

/// Generate a reference to this DIType. Uses the type identifier instead
/// of the actual MDNode if possible, to help type uniquing.
DIScopeRef DIScope::getRef() const {
  if (!isCompositeType())
    return DIScopeRef(*this);
  DICompositeType DTy(DbgNode);
  if (!DTy.getIdentifier())
    return DIScopeRef(*this);
  return DIScopeRef(DTy.getIdentifier());
}

/// \brief Set the containing type.
void DICompositeType::setContainingType(DICompositeType ContainingType) {
  TrackingVH<MDNode> N(*this);
  N->replaceOperandWith(12, ContainingType.getRef());
  DbgNode = N;
}

/// isInlinedFnArgument - Return true if this variable provides debugging
/// information for an inlined function arguments.
bool DIVariable::isInlinedFnArgument(const Function *CurFn) {
  assert(CurFn && "Invalid function");
  if (!getContext().isSubprogram())
    return false;
  // This variable is not inlined function argument if its scope
  // does not describe current function.
  return !DISubprogram(getContext()).describes(CurFn);
}

/// describes - Return true if this subprogram provides debugging
/// information for the function F.
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

unsigned DISubprogram::isOptimized() const {
  assert(DbgNode && "Invalid subprogram descriptor!");
  if (DbgNode->getNumOperands() == 15)
    return getUnsignedField(14);
  return 0;
}

MDNode *DISubprogram::getVariablesNodes() const {
  return getNodeField(DbgNode, 18);
}

DIArray DISubprogram::getVariables() const {
  return DIArray(getNodeField(DbgNode, 18));
}

Value *DITemplateValueParameter::getValue() const {
  return getField(DbgNode, 4);
}

// If the current node has a parent scope then return that,
// else return an empty scope.
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
  return DIScopeRef(NULL);
}

// If the scope node has a name, return that, else return an empty string.
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
  if (!DbgNode || DbgNode->getNumOperands() < 13)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 7));
}

DIArray DICompileUnit::getRetainedTypes() const {
  if (!DbgNode || DbgNode->getNumOperands() < 13)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 8));
}

DIArray DICompileUnit::getSubprograms() const {
  if (!DbgNode || DbgNode->getNumOperands() < 13)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 9));
}

DIArray DICompileUnit::getGlobalVariables() const {
  if (!DbgNode || DbgNode->getNumOperands() < 13)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 10));
}

DIArray DICompileUnit::getImportedEntities() const {
  if (!DbgNode || DbgNode->getNumOperands() < 13)
    return DIArray();

  return DIArray(getNodeField(DbgNode, 11));
}

/// fixupSubprogramName - Replace contains special characters used
/// in a typical Objective-C names with '.' in a given string.
static void fixupSubprogramName(DISubprogram Fn, SmallVectorImpl<char> &Out) {
  StringRef FName =
      Fn.getFunction() ? Fn.getFunction()->getName() : Fn.getName();
  FName = Function::getRealLinkageName(FName);

  StringRef Prefix("llvm.dbg.lv.");
  Out.reserve(FName.size() + Prefix.size());
  Out.append(Prefix.begin(), Prefix.end());

  bool isObjCLike = false;
  for (size_t i = 0, e = FName.size(); i < e; ++i) {
    char C = FName[i];
    if (C == '[')
      isObjCLike = true;

    if (isObjCLike && (C == '[' || C == ']' || C == ' ' || C == ':' ||
                       C == '+' || C == '(' || C == ')'))
      Out.push_back('.');
    else
      Out.push_back(C);
  }
}

/// getFnSpecificMDNode - Return a NameMDNode, if available, that is
/// suitable to hold function specific information.
NamedMDNode *llvm::getFnSpecificMDNode(const Module &M, DISubprogram Fn) {
  SmallString<32> Name;
  fixupSubprogramName(Fn, Name);
  return M.getNamedMetadata(Name.str());
}

/// getOrInsertFnSpecificMDNode - Return a NameMDNode that is suitable
/// to hold function specific information.
NamedMDNode *llvm::getOrInsertFnSpecificMDNode(Module &M, DISubprogram Fn) {
  SmallString<32> Name;
  fixupSubprogramName(Fn, Name);
  return M.getOrInsertNamedMetadata(Name.str());
}

/// createInlinedVariable - Create a new inlined variable based on current
/// variable.
/// @param DV            Current Variable.
/// @param InlinedScope  Location at current variable is inlined.
DIVariable llvm::createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                       LLVMContext &VMContext) {
  SmallVector<Value *, 16> Elts;
  // Insert inlined scope as 7th element.
  for (unsigned i = 0, e = DV->getNumOperands(); i != e; ++i)
    i == 7 ? Elts.push_back(InlinedScope) : Elts.push_back(DV->getOperand(i));
  return DIVariable(MDNode::get(VMContext, Elts));
}

/// cleanseInlinedVariable - Remove inlined scope from the variable.
DIVariable llvm::cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext) {
  SmallVector<Value *, 16> Elts;
  // Insert inlined scope as 7th element.
  for (unsigned i = 0, e = DV->getNumOperands(); i != e; ++i)
    i == 7 ? Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext)))
           : Elts.push_back(DV->getOperand(i));
  return DIVariable(MDNode::get(VMContext, Elts));
}

/// getDISubprogram - Find subprogram that is enclosing this scope.
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

/// getDICompositeType - Find underlying composite type.
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

/// Update DITypeIdentifierMap by going through retained types of each CU.
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

/// processModule - Process entire module and collect debug info.
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
          processType(DIG.getType());
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
        DIDescriptor Entity = Import.getEntity();
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

/// processLocation - Process DILocation.
void DebugInfoFinder::processLocation(const Module &M, DILocation Loc) {
  if (!Loc)
    return;
  InitializeTypeMap(M);
  processScope(Loc.getScope());
  processLocation(M, Loc.getOrigLocation());
}

/// processType - Process DIType.
void DebugInfoFinder::processType(DIType DT) {
  if (!addType(DT))
    return;
  processScope(DT.getContext().resolve(TypeIdentifierMap));
  if (DT.isCompositeType()) {
    DICompositeType DCT(DT);
    processType(DCT.getTypeDerivedFrom().resolve(TypeIdentifierMap));
    DIArray DA = DCT.getTypeArray();
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

/// processLexicalBlock
void DebugInfoFinder::processLexicalBlock(DILexicalBlock LB) {
  DIScope Context = LB.getContext();
  if (Context.isLexicalBlock())
    return processLexicalBlock(DILexicalBlock(Context));
  else if (Context.isLexicalBlockFile()) {
    DILexicalBlockFile DBF = DILexicalBlockFile(Context);
    return processLexicalBlock(DILexicalBlock(DBF.getScope()));
  } else
    return processSubprogram(DISubprogram(Context));
}

/// processSubprogram - Process DISubprogram.
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

/// processDeclare - Process DbgDeclareInst.
void DebugInfoFinder::processDeclare(const Module &M,
                                     const DbgDeclareInst *DDI) {
  MDNode *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV))
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType());
}

void DebugInfoFinder::processValue(const Module &M, const DbgValueInst *DVI) {
  MDNode *N = dyn_cast<MDNode>(DVI->getVariable());
  if (!N)
    return;
  InitializeTypeMap(M);

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV))
    return;
  processScope(DIVariable(N).getContext());
  processType(DIVariable(N).getType());
}

/// addType - Add type into Tys.
bool DebugInfoFinder::addType(DIType DT) {
  if (!DT)
    return false;

  if (!NodesSeen.insert(DT))
    return false;

  TYs.push_back(DT);
  return true;
}

/// addCompileUnit - Add compile unit into CUs.
bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (!CU)
    return false;
  if (!NodesSeen.insert(CU))
    return false;

  CUs.push_back(CU);
  return true;
}

/// addGlobalVariable - Add global variable into GVs.
bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (!DIG)
    return false;

  if (!NodesSeen.insert(DIG))
    return false;

  GVs.push_back(DIG);
  return true;
}

// addSubprogram - Add subprgoram into SPs.
bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (!SP)
    return false;

  if (!NodesSeen.insert(SP))
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
  if (!NodesSeen.insert(Scope))
    return false;
  Scopes.push_back(Scope);
  return true;
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//

/// dump - Print descriptor to dbgs() with a newline.
void DIDescriptor::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

/// print - Print descriptor.
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
}

void DIDerivedType::printInternal(raw_ostream &OS) const {
  DIType::printInternal(OS);
  OS << " [from " << getTypeDerivedFrom().getName() << ']';
}

void DICompositeType::printInternal(raw_ostream &OS) const {
  DIType::printInternal(OS);
  DIArray A = getTypeArray();
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

/// Specialize constructor to make sure it has the correct type.
template <> DIRef<DIScope>::DIRef(const Value *V) : Val(V) {
  assert(isScopeRef(V) && "DIScopeRef should be a MDString or MDNode");
}
template <> DIRef<DIType>::DIRef(const Value *V) : Val(V) {
  assert(isTypeRef(V) && "DITypeRef should be a MDString or MDNode");
}

/// Specialize getFieldAs to handle fields that are references to DIScopes.
template <>
DIScopeRef DIDescriptor::getFieldAs<DIScopeRef>(unsigned Elt) const {
  return DIScopeRef(getField(DbgNode, Elt));
}
/// Specialize getFieldAs to handle fields that are references to DITypes.
template <> DITypeRef DIDescriptor::getFieldAs<DITypeRef>(unsigned Elt) const {
  return DITypeRef(getField(DbgNode, Elt));
}

/// Strip debug info in the module if it exists.
/// To do this, we remove all calls to the debugger intrinsics and any named
/// metadata for debugging. We also remove debug locations for instructions.
/// Return true if module is modified.
bool llvm::StripDebugInfo(Module &M) {

  bool Changed = false;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (Function *Declare = M.getFunction("llvm.dbg.declare")) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->use_back());
      CI->eraseFromParent();
    }
    Declare->eraseFromParent();
    Changed = true;
  }

  if (Function *DbgVal = M.getFunction("llvm.dbg.value")) {
    while (!DbgVal->use_empty()) {
      CallInst *CI = cast<CallInst>(DbgVal->use_back());
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

/// Return Debug Info Metadata Version by checking module flags.
unsigned llvm::getDebugMetadataVersionFromModule(const Module &M) {
  Value *Val = M.getModuleFlag("Debug Info Version");
  if (!Val)
    return 0;
  return cast<ConstantInt>(Val)->getZExtValue();
}
