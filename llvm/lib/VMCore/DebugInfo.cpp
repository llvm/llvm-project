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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::dwarf;

//===----------------------------------------------------------------------===//
// DIDescriptor
//===----------------------------------------------------------------------===//

DIDescriptor::DIDescriptor(const DIFile F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DISubprogram F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DILexicalBlockFile F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DILexicalBlock F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DIVariable F) : DbgNode(F.DbgNode) {
}

DIDescriptor::DIDescriptor(const DIType F) : DbgNode(F.DbgNode) {
}

StringRef
DIDescriptor::getStringField(unsigned Elt) const {
  if (DbgNode == 0)
    return StringRef();

  if (Elt < DbgNode->getNumOperands())
    if (MDString *MDS = dyn_cast_or_null<MDString>(DbgNode->getOperand(Elt)))
      return MDS->getString();

  return StringRef();
}

uint64_t DIDescriptor::getUInt64Field(unsigned Elt) const {
  if (DbgNode == 0)
    return 0;

  if (Elt < DbgNode->getNumOperands())
    if (ConstantInt *CI = dyn_cast_or_null<ConstantInt>(DbgNode->getOperand(Elt)))
      return CI->getZExtValue();

  return 0;
}

DIDescriptor DIDescriptor::getDescriptorField(unsigned Elt) const {
  if (DbgNode == 0)
    return DIDescriptor();

  if (Elt < DbgNode->getNumOperands())
    return
      DIDescriptor(dyn_cast_or_null<const MDNode>(DbgNode->getOperand(Elt)));
  return DIDescriptor();
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
    MDNode *Node = const_cast<MDNode*>(DbgNode);
    Node->replaceOperandWith(Elt, F);
  }
}

unsigned DIVariable::getNumAddrElements() const {
  if (getVersion() <= LLVMDebugVersion8)
    return DbgNode->getNumOperands()-6;
  if (getVersion() == LLVMDebugVersion9)
    return DbgNode->getNumOperands()-7;
  return DbgNode->getNumOperands()-8;
}

/// getInlinedAt - If this variable is inlined then return inline location.
MDNode *DIVariable::getInlinedAt() const {
  if (getVersion() <= LLVMDebugVersion9)
    return NULL;
  return dyn_cast_or_null<MDNode>(DbgNode->getOperand(7));
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// isBasicType - Return true if the specified tag is legal for
/// DIBasicType.
bool DIDescriptor::isBasicType() const {
  if (!DbgNode) return false;
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
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_pointer_type:
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
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_class_type:
    return true;
  default:
    return false;
  }
}

/// isVariable - Return true if the specified tag is legal for DIVariable.
bool DIDescriptor::isVariable() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_auto_variable:
  case dwarf::DW_TAG_arg_variable:
  case dwarf::DW_TAG_return_variable:
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

/// isGlobal - Return true if the specified tag is legal for DIGlobal.
bool DIDescriptor::isGlobal() const {
  return isGlobalVariable();
}

/// isUnspecifiedParmeter - Return true if the specified tag is
/// DW_TAG_unspecified_parameters.
bool DIDescriptor::isUnspecifiedParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_unspecified_parameters;
}

/// isScope - Return true if the specified tag is one of the scope
/// related tag.
bool DIDescriptor::isScope() const {
  if (!DbgNode) return false;
  switch (getTag()) {
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_namespace:
    return true;
  default:
    break;
  }
  return false;
}

/// isTemplateTypeParameter - Return true if the specified tag is
/// DW_TAG_template_type_parameter.
bool DIDescriptor::isTemplateTypeParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_type_parameter;
}

/// isTemplateValueParameter - Return true if the specified tag is
/// DW_TAG_template_value_parameter.
bool DIDescriptor::isTemplateValueParameter() const {
  return DbgNode && getTag() == dwarf::DW_TAG_template_value_parameter;
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

/// isObjCProperty - Return true if the specified tag is DW_TAG
bool DIDescriptor::isObjCProperty() const {
  return DbgNode && getTag() == dwarf::DW_TAG_APPLE_property;
}
//===----------------------------------------------------------------------===//
// Simple Descriptor Constructors and other Methods
//===----------------------------------------------------------------------===//

DIType::DIType(const MDNode *N) : DIScope(N) {
  if (!N) return;
  if (!isBasicType() && !isDerivedType() && !isCompositeType()) {
    DbgNode = 0;
  }
}

unsigned DIArray::getNumElements() const {
  if (!DbgNode)
    return 0;
  return DbgNode->getNumOperands();
}

/// replaceAllUsesWith - Replace all uses of debug info referenced by
/// this descriptor.
void DIType::replaceAllUsesWith(DIDescriptor &D) {
  if (!DbgNode)
    return;

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode*>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value*>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// replaceAllUsesWith - Replace all uses of debug info referenced by
/// this descriptor.
void DIType::replaceAllUsesWith(MDNode *D) {
  if (!DbgNode)
    return;

  // Since we use a TrackingVH for the node, its easy for clients to manufacture
  // legitimate situations where they want to replaceAllUsesWith() on something
  // which, due to uniquing, has merged with the source. We shield clients from
  // this detail by allowing a value to be replaced with replaceAllUsesWith()
  // itself.
  if (DbgNode != D) {
    MDNode *Node = const_cast<MDNode*>(DbgNode);
    const MDNode *DN = D;
    const Value *V = cast_or_null<Value>(DN);
    Node->replaceAllUsesWith(const_cast<Value*>(V));
    MDNode::deleteTemporary(Node);
  }
}

/// isUnsignedDIType - Return true if type encoding is unsigned.
bool DIType::isUnsignedDIType() {
  DIDerivedType DTy(DbgNode);
  if (DTy.Verify())
    return DTy.getTypeDerivedFrom().isUnsignedDIType();

  DIBasicType BTy(DbgNode);
  if (BTy.Verify()) {
    unsigned Encoding = BTy.getEncoding();
    if (Encoding == dwarf::DW_ATE_unsigned ||
        Encoding == dwarf::DW_ATE_unsigned_char)
      return true;
  }
  return false;
}

/// Verify - Verify that a compile unit is well formed.
bool DICompileUnit::Verify() const {
  if (!DbgNode)
    return false;
  StringRef N = getFilename();
  if (N.empty())
    return false;
  // It is possible that directory and produce string is empty.
  return true;
}

/// Verify - Verify that an ObjC property is well formed.
bool DIObjCProperty::Verify() const {
  if (!DbgNode)
    return false;
  unsigned Tag = getTag();
  if (Tag != dwarf::DW_TAG_APPLE_property) return false;
  DIType Ty = getType();
  if (!Ty.Verify()) return false;

  // Don't worry about the rest of the strings for now.
  return true;
}

/// Verify - Verify that a type descriptor is well formed.
bool DIType::Verify() const {
  if (!DbgNode)
    return false;
  if (getContext() && !getContext().Verify())
    return false;
  unsigned Tag = getTag();
  if (!isBasicType() && Tag != dwarf::DW_TAG_const_type &&
      Tag != dwarf::DW_TAG_volatile_type && Tag != dwarf::DW_TAG_pointer_type &&
      Tag != dwarf::DW_TAG_reference_type &&
      Tag != dwarf::DW_TAG_rvalue_reference_type &&
      Tag != dwarf::DW_TAG_restrict_type && Tag != dwarf::DW_TAG_vector_type &&
      Tag != dwarf::DW_TAG_array_type &&
      Tag != dwarf::DW_TAG_enumeration_type &&
      Tag != dwarf::DW_TAG_subroutine_type &&
      getFilename().empty())
    return false;
  return true;
}

/// Verify - Verify that a basic type descriptor is well formed.
bool DIBasicType::Verify() const {
  return isBasicType();
}

/// Verify - Verify that a derived type descriptor is well formed.
bool DIDerivedType::Verify() const {
  return isDerivedType();
}

/// Verify - Verify that a composite type descriptor is well formed.
bool DICompositeType::Verify() const {
  if (!DbgNode)
    return false;
  if (getContext() && !getContext().Verify())
    return false;

  return true;
}

/// Verify - Verify that a subprogram descriptor is well formed.
bool DISubprogram::Verify() const {
  if (!DbgNode)
    return false;

  if (getContext() && !getContext().Verify())
    return false;

  DICompositeType Ty = getType();
  if (!Ty.Verify())
    return false;
  return true;
}

/// Verify - Verify that a global variable descriptor is well formed.
bool DIGlobalVariable::Verify() const {
  if (!DbgNode)
    return false;

  if (getDisplayName().empty())
    return false;

  if (getContext() && !getContext().Verify())
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  if (!getGlobal() && !getConstant())
    return false;

  return true;
}

/// Verify - Verify that a variable descriptor is well formed.
bool DIVariable::Verify() const {
  if (!DbgNode)
    return false;

  if (getContext() && !getContext().Verify())
    return false;

  DIType Ty = getType();
  if (!Ty.Verify())
    return false;

  return true;
}

/// Verify - Verify that a location descriptor is well formed.
bool DILocation::Verify() const {
  if (!DbgNode)
    return false;

  return DbgNode->getNumOperands() == 4;
}

/// Verify - Verify that a namespace descriptor is well formed.
bool DINameSpace::Verify() const {
  if (!DbgNode)
    return false;
  if (getName().empty())
    return false;
  return true;
}

/// getOriginalTypeSize - If this type is derived from a base type then
/// return base type size.
uint64_t DIDerivedType::getOriginalTypeSize() const {
  unsigned Tag = getTag();

  if (Tag != dwarf::DW_TAG_member && Tag != dwarf::DW_TAG_typedef &&
      Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type)
    return getSizeInBits();

  DIType BaseType = getTypeDerivedFrom();

  // If this type is not derived from any type then take conservative approach.
  if (!BaseType.isValid())
    return getSizeInBits();

  // If this is a derived type, go ahead and get the base type, unless it's a
  // reference then it's just the size of the field. Pointer types have no need
  // of this since they're a different type of qualification on the type.
  if (BaseType.getTag() == dwarf::DW_TAG_reference_type ||
      BaseType.getTag() == dwarf::DW_TAG_rvalue_reference_type)
    return getSizeInBits();

  if (BaseType.isDerivedType())
    return DIDerivedType(BaseType).getOriginalTypeSize();

  return BaseType.getSizeInBits();
}

/// getObjCProperty - Return property node, if this ivar is associated with one.
MDNode *DIDerivedType::getObjCProperty() const {
  if (getVersion() <= LLVMDebugVersion11 || DbgNode->getNumOperands() <= 10)
    return NULL;
  return dyn_cast_or_null<MDNode>(DbgNode->getOperand(10));
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
  assert (DbgNode && "Invalid subprogram descriptor!");
  if (DbgNode->getNumOperands() == 16)
    return getUnsignedField(15);
  return 0;
}

MDNode *DISubprogram::getVariablesNodes() const {
  if (!DbgNode || DbgNode->getNumOperands() <= 19)
    return NULL;
  if (MDNode *Temp = dyn_cast_or_null<MDNode>(DbgNode->getOperand(19)))
    return dyn_cast_or_null<MDNode>(Temp->getOperand(0));
  return NULL;
}

DIArray DISubprogram::getVariables() const {
  if (!DbgNode || DbgNode->getNumOperands() <= 19)
    return DIArray();
  if (MDNode *T = dyn_cast_or_null<MDNode>(DbgNode->getOperand(19)))
    if (MDNode *A = dyn_cast_or_null<MDNode>(T->getOperand(0)))
      return DIArray(A);
  return DIArray();
}

StringRef DIScope::getFilename() const {
  if (!DbgNode)
    return StringRef();
  if (isLexicalBlockFile())
    return DILexicalBlockFile(DbgNode).getFilename();
  if (isLexicalBlock())
    return DILexicalBlock(DbgNode).getFilename();
  if (isSubprogram())
    return DISubprogram(DbgNode).getFilename();
  if (isCompileUnit())
    return DICompileUnit(DbgNode).getFilename();
  if (isNameSpace())
    return DINameSpace(DbgNode).getFilename();
  if (isType())
    return DIType(DbgNode).getFilename();
  if (isFile())
    return DIFile(DbgNode).getFilename();
  llvm_unreachable("Invalid DIScope!");
}

StringRef DIScope::getDirectory() const {
  if (!DbgNode)
    return StringRef();
  if (isLexicalBlockFile())
    return DILexicalBlockFile(DbgNode).getDirectory();
  if (isLexicalBlock())
    return DILexicalBlock(DbgNode).getDirectory();
  if (isSubprogram())
    return DISubprogram(DbgNode).getDirectory();
  if (isCompileUnit())
    return DICompileUnit(DbgNode).getDirectory();
  if (isNameSpace())
    return DINameSpace(DbgNode).getDirectory();
  if (isType())
    return DIType(DbgNode).getDirectory();
  if (isFile())
    return DIFile(DbgNode).getDirectory();
  llvm_unreachable("Invalid DIScope!");
}

DIArray DICompileUnit::getEnumTypes() const {
  if (!DbgNode || DbgNode->getNumOperands() < 14)
    return DIArray();

  if (MDNode *N = dyn_cast_or_null<MDNode>(DbgNode->getOperand(10)))
    if (MDNode *A = dyn_cast_or_null<MDNode>(N->getOperand(0)))
      return DIArray(A);
  return DIArray();
}

DIArray DICompileUnit::getRetainedTypes() const {
  if (!DbgNode || DbgNode->getNumOperands() < 14)
    return DIArray();

  if (MDNode *N = dyn_cast_or_null<MDNode>(DbgNode->getOperand(11)))
    if (MDNode *A = dyn_cast_or_null<MDNode>(N->getOperand(0)))
      return DIArray(A);
  return DIArray();
}

DIArray DICompileUnit::getSubprograms() const {
  if (!DbgNode || DbgNode->getNumOperands() < 14)
    return DIArray();

  if (MDNode *N = dyn_cast_or_null<MDNode>(DbgNode->getOperand(12)))
    if (MDNode *A = dyn_cast_or_null<MDNode>(N->getOperand(0)))
      return DIArray(A);
  return DIArray();
}


DIArray DICompileUnit::getGlobalVariables() const {
  if (!DbgNode || DbgNode->getNumOperands() < 14)
    return DIArray();

  if (MDNode *N = dyn_cast_or_null<MDNode>(DbgNode->getOperand(13)))
    if (MDNode *A = dyn_cast_or_null<MDNode>(N->getOperand(0)))
      return DIArray(A);
  return DIArray();
}

/// fixupObjcLikeName - Replace contains special characters used
/// in a typical Objective-C names with '.' in a given string.
static void fixupObjcLikeName(StringRef Str, SmallVectorImpl<char> &Out) {
  bool isObjCLike = false;
  for (size_t i = 0, e = Str.size(); i < e; ++i) {
    char C = Str[i];
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
  SmallString<32> Name = StringRef("llvm.dbg.lv.");
  StringRef FName = "fn";
  if (Fn.getFunction())
    FName = Fn.getFunction()->getName();
  else
    FName = Fn.getName();
  char One = '\1';
  if (FName.startswith(StringRef(&One, 1)))
    FName = FName.substr(1);
  fixupObjcLikeName(FName, Name);
  return M.getNamedMetadata(Name.str());
}

/// getOrInsertFnSpecificMDNode - Return a NameMDNode that is suitable
/// to hold function specific information.
NamedMDNode *llvm::getOrInsertFnSpecificMDNode(Module &M, DISubprogram Fn) {
  SmallString<32> Name = StringRef("llvm.dbg.lv.");
  StringRef FName = "fn";
  if (Fn.getFunction())
    FName = Fn.getFunction()->getName();
  else
    FName = Fn.getName();
  char One = '\1';
  if (FName.startswith(StringRef(&One, 1)))
    FName = FName.substr(1);
  fixupObjcLikeName(FName, Name);
  
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
    i == 7 ? Elts.push_back(InlinedScope) :
             Elts.push_back(DV->getOperand(i));
  return DIVariable(MDNode::get(VMContext, Elts));
}

/// cleanseInlinedVariable - Remove inlined scope from the variable.
DIVariable llvm::cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext) {
  SmallVector<Value *, 16> Elts;
  // Insert inlined scope as 7th element.
  for (unsigned i = 0, e = DV->getNumOperands(); i != e; ++i)
    i == 7 ? 
      Elts.push_back(Constant::getNullValue(Type::getInt32Ty(VMContext))):
      Elts.push_back(DV->getOperand(i));
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

  if (T.isDerivedType())
    return getDICompositeType(DIDerivedType(T).getTypeDerivedFrom());

  return DICompositeType();
}

/// isSubprogramContext - Return true if Context is either a subprogram
/// or another context nested inside a subprogram.
bool llvm::isSubprogramContext(const MDNode *Context) {
  if (!Context)
    return false;
  DIDescriptor D(Context);
  if (D.isSubprogram())
    return true;
  if (D.isType())
    return isSubprogramContext(DIType(Context).getContext());
  return false;
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

/// processModule - Process entire module and collect debug info.
void DebugInfoFinder::processModule(Module &M) {
  if (NamedMDNode *CU_Nodes = M.getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
      DICompileUnit CU(CU_Nodes->getOperand(i));
      addCompileUnit(CU);
      if (CU.getVersion() > LLVMDebugVersion10) {
        DIArray GVs = CU.getGlobalVariables();
        for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i) {
          DIGlobalVariable DIG(GVs.getElement(i));
          if (addGlobalVariable(DIG))
            processType(DIG.getType());
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
        return;
      }
    }
  }

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (Function::iterator FI = (*I).begin(), FE = (*I).end(); FI != FE; ++FI)
      for (BasicBlock::iterator BI = (*FI).begin(), BE = (*FI).end(); BI != BE;
           ++BI) {
        if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI))
          processDeclare(DDI);

        DebugLoc Loc = BI->getDebugLoc();
        if (Loc.isUnknown())
          continue;

        LLVMContext &Ctx = BI->getContext();
        DIDescriptor Scope(Loc.getScope(Ctx));

        if (Scope.isCompileUnit())
          addCompileUnit(DICompileUnit(Scope));
        else if (Scope.isSubprogram())
          processSubprogram(DISubprogram(Scope));
        else if (Scope.isLexicalBlockFile()) {
          DILexicalBlockFile DBF = DILexicalBlockFile(Scope);
          processLexicalBlock(DILexicalBlock(DBF.getScope()));
        }
        else if (Scope.isLexicalBlock())
          processLexicalBlock(DILexicalBlock(Scope));

        if (MDNode *IA = Loc.getInlinedAt(Ctx))
          processLocation(DILocation(IA));
      }

  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.gv")) {
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      DIGlobalVariable DIG(cast<MDNode>(NMD->getOperand(i)));
      if (addGlobalVariable(DIG)) {
        if (DIG.getVersion() <= LLVMDebugVersion10)
          addCompileUnit(DIG.getCompileUnit());
        processType(DIG.getType());
      }
    }
  }

  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.sp"))
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      processSubprogram(DISubprogram(NMD->getOperand(i)));
}

/// processLocation - Process DILocation.
void DebugInfoFinder::processLocation(DILocation Loc) {
  if (!Loc.Verify()) return;
  DIDescriptor S(Loc.getScope());
  if (S.isCompileUnit())
    addCompileUnit(DICompileUnit(S));
  else if (S.isSubprogram())
    processSubprogram(DISubprogram(S));
  else if (S.isLexicalBlock())
    processLexicalBlock(DILexicalBlock(S));
  else if (S.isLexicalBlockFile()) {
    DILexicalBlockFile DBF = DILexicalBlockFile(S);
    processLexicalBlock(DILexicalBlock(DBF.getScope()));
  }
  processLocation(Loc.getOrigLocation());
}

/// processType - Process DIType.
void DebugInfoFinder::processType(DIType DT) {
  if (!addType(DT))
    return;
  if (DT.getVersion() <= LLVMDebugVersion10)
    addCompileUnit(DT.getCompileUnit());
  if (DT.isCompositeType()) {
    DICompositeType DCT(DT);
    processType(DCT.getTypeDerivedFrom());
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
    processType(DDT.getTypeDerivedFrom());
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
  }
  else
    return processSubprogram(DISubprogram(Context));
}

/// processSubprogram - Process DISubprogram.
void DebugInfoFinder::processSubprogram(DISubprogram SP) {
  if (!addSubprogram(SP))
    return;
  if (SP.getVersion() <= LLVMDebugVersion10)
    addCompileUnit(SP.getCompileUnit());
  processType(SP.getType());
}

/// processDeclare - Process DbgDeclareInst.
void DebugInfoFinder::processDeclare(DbgDeclareInst *DDI) {
  MDNode *N = dyn_cast<MDNode>(DDI->getVariable());
  if (!N) return;

  DIDescriptor DV(N);
  if (!DV.isVariable())
    return;

  if (!NodesSeen.insert(DV))
    return;
  if (DIVariable(N).getVersion() <= LLVMDebugVersion10)
    addCompileUnit(DIVariable(N).getCompileUnit());
  processType(DIVariable(N).getType());
}

/// addType - Add type into Tys.
bool DebugInfoFinder::addType(DIType DT) {
  if (!DT.isValid())
    return false;

  if (!NodesSeen.insert(DT))
    return false;

  TYs.push_back(DT);
  return true;
}

/// addCompileUnit - Add compile unit into CUs.
bool DebugInfoFinder::addCompileUnit(DICompileUnit CU) {
  if (!CU.Verify())
    return false;

  if (!NodesSeen.insert(CU))
    return false;

  CUs.push_back(CU);
  return true;
}

/// addGlobalVariable - Add global variable into GVs.
bool DebugInfoFinder::addGlobalVariable(DIGlobalVariable DIG) {
  if (!DIDescriptor(DIG).isGlobalVariable())
    return false;

  if (!NodesSeen.insert(DIG))
    return false;

  GVs.push_back(DIG);
  return true;
}

// addSubprogram - Add subprgoram into SPs.
bool DebugInfoFinder::addSubprogram(DISubprogram SP) {
  if (!DIDescriptor(SP).isSubprogram())
    return false;

  if (!NodesSeen.insert(SP))
    return false;

  SPs.push_back(SP);
  return true;
}

//===----------------------------------------------------------------------===//
// DIDescriptor: dump routines for all descriptors.
//===----------------------------------------------------------------------===//

/// dump - Print descriptor to dbgs() with a newline.
void DIDescriptor::dump() const {
  print(dbgs()); dbgs() << '\n';
}

/// print - Print descriptor.
void DIDescriptor::print(raw_ostream &OS) const {
  if (!DbgNode) return;

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
  } else if (this->isScope()) {
    DIScope(DbgNode).printInternal(OS);
  }
}

void DISubrange::printInternal(raw_ostream &OS) const {
  OS << " [" << getLo() << ", " << getHi() << ']';
}

void DIScope::printInternal(raw_ostream &OS) const {
  OS << " [" << getDirectory() << "/" << getFilename() << ']';
}

void DICompileUnit::printInternal(raw_ostream &OS) const {
  DIScope::printInternal(OS);
  if (unsigned Lang = getLanguage())
    OS << " [" << dwarf::LanguageString(Lang) << ']';
}

void DIEnumerator::printInternal(raw_ostream &OS) const {
  OS << " [" << getName() << " :: " << getEnumValue() << ']';
}

void DIType::printInternal(raw_ostream &OS) const {
  if (!DbgNode) return;

  StringRef Res = getName();
  if (!Res.empty())
    OS << " [" << Res << "]";

  // TODO: Print context?

  OS << " [line " << getLineNumber()
     << ", size " << getSizeInBits()
     << ", align " << getAlignInBits()
     << ", offset " << getOffsetInBits();
  if (isBasicType())
    if (const char *Enc = 
        dwarf::AttributeEncodingString(DIBasicType(DbgNode).getEncoding()))
      OS << ", enc " << Enc;
  OS << "]";

  if (isPrivate())
    OS << " [private]";
  else if (isProtected())
    OS << " [protected]";

  if (isForwardDecl())
    OS << " [fwd]";
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

void DISubprogram::printInternal(raw_ostream &OS) const {
  // TODO : Print context
  OS << " [line " << getLineNumber() << ']';

  if (isLocalToUnit())
    OS << " [local]";

  if (isDefinition())
    OS << " [def]";

  if (getScopeLineNumber() != getLineNumber())
    OS << " [scope " << getScopeLineNumber() << "]";

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

  OS << " [line " << getLineNumber()
     << ", properties " << getUnsignedField(6) << ']';
}

static void printDebugLoc(DebugLoc DL, raw_ostream &CommentOS,
                          const LLVMContext &Ctx) {
  if (!DL.isUnknown()) {          // Print source line info.
    DIScope Scope(DL.getScope(Ctx));
    // Omit the directory, because it's likely to be long and uninteresting.
    if (Scope.Verify())
      CommentOS << Scope.getFilename();
    else
      CommentOS << "<unknown>";
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
