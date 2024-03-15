//===- DebugImporter.cpp - LLVM to MLIR Debug conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugImporter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

Location DebugImporter::translateFuncLocation(llvm::Function *func) {
  llvm::DISubprogram *subprogram = func->getSubprogram();
  if (!subprogram)
    return UnknownLoc::get(context);

  // Add a fused location to link the subprogram information.
  StringAttr funcName = StringAttr::get(context, subprogram->getName());
  StringAttr fileName = StringAttr::get(context, subprogram->getFilename());
  return FusedLocWith<DISubprogramAttr>::get(
      {NameLoc::get(funcName),
       FileLineColLoc::get(fileName, subprogram->getLine(), /*column=*/0)},
      translate(subprogram), context);
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

DIBasicTypeAttr DebugImporter::translateImpl(llvm::DIBasicType *node) {
  return DIBasicTypeAttr::get(context, node->getTag(), node->getName(),
                              node->getSizeInBits(), node->getEncoding());
}

DICompileUnitAttr DebugImporter::translateImpl(llvm::DICompileUnit *node) {
  std::optional<DIEmissionKind> emissionKind =
      symbolizeDIEmissionKind(node->getEmissionKind());
  return DICompileUnitAttr::get(
      context, DistinctAttr::create(UnitAttr::get(context)),
      node->getSourceLanguage(), translate(node->getFile()),
      getStringAttrOrNull(node->getRawProducer()), node->isOptimized(),
      emissionKind.value());
}

DICompositeTypeAttr DebugImporter::translateImpl(llvm::DICompositeType *node) {
  std::optional<DIFlags> flags = symbolizeDIFlags(node->getFlags());
  SmallVector<DINodeAttr> elements;
  for (llvm::DINode *element : node->getElements()) {
    assert(element && "expected a non-null element type");
    elements.push_back(translate(element));
  }
  // Drop the elements parameter if a cyclic dependency is detected. We
  // currently cannot model these cycles and thus drop the parameter if
  // required. A cyclic dependency is detected if one of the element nodes
  // translates to a nullptr since the node is already on the translation stack.
  // TODO: Support debug metadata with cyclic dependencies.
  if (llvm::is_contained(elements, nullptr))
    elements.clear();
  DITypeAttr baseType = translate(node->getBaseType());
  // Arrays require a base type, otherwise the debug metadata is considered to
  // be malformed.
  if (node->getTag() == llvm::dwarf::DW_TAG_array_type && !baseType)
    return nullptr;
  return DICompositeTypeAttr::get(
      context, node->getTag(), getStringAttrOrNull(node->getRawName()),
      translate(node->getFile()), node->getLine(), translate(node->getScope()),
      baseType, flags.value_or(DIFlags::Zero), node->getSizeInBits(),
      node->getAlignInBits(), elements);
}

DIDerivedTypeAttr DebugImporter::translateImpl(llvm::DIDerivedType *node) {
  // Return nullptr if the base type is a cyclic dependency.
  DITypeAttr baseType = translate(node->getBaseType());
  if (node->getBaseType() && !baseType)
    return nullptr;
  return DIDerivedTypeAttr::get(
      context, node->getTag(), getStringAttrOrNull(node->getRawName()),
      baseType, node->getSizeInBits(), node->getAlignInBits(),
      node->getOffsetInBits());
}

DIFileAttr DebugImporter::translateImpl(llvm::DIFile *node) {
  return DIFileAttr::get(context, node->getFilename(), node->getDirectory());
}

DILabelAttr DebugImporter::translateImpl(llvm::DILabel *node) {
  // Return nullptr if the scope or type is a cyclic dependency.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  return DILabelAttr::get(context, scope,
                          getStringAttrOrNull(node->getRawName()),
                          translate(node->getFile()), node->getLine());
}

DILexicalBlockAttr DebugImporter::translateImpl(llvm::DILexicalBlock *node) {
  // Return nullptr if the scope or type is a cyclic dependency.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  return DILexicalBlockAttr::get(context, scope, translate(node->getFile()),
                                 node->getLine(), node->getColumn());
}

DILexicalBlockFileAttr
DebugImporter::translateImpl(llvm::DILexicalBlockFile *node) {
  // Return nullptr if the scope or type is a cyclic dependency.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  return DILexicalBlockFileAttr::get(context, scope, translate(node->getFile()),
                                     node->getDiscriminator());
}

DIGlobalVariableAttr
DebugImporter::translateImpl(llvm::DIGlobalVariable *node) {
  // Names of DIGlobalVariables can be empty. MLIR models them as null, instead
  // of empty strings, so this special handling is necessary.
  auto convertToStringAttr = [&](StringRef name) -> StringAttr {
    if (name.empty())
      return {};
    return StringAttr::get(context, node->getName());
  };
  return DIGlobalVariableAttr::get(
      context, translate(node->getScope()),
      convertToStringAttr(node->getName()),
      convertToStringAttr(node->getLinkageName()), translate(node->getFile()),
      node->getLine(), translate(node->getType()), node->isLocalToUnit(),
      node->isDefinition(), node->getAlignInBits());
}

DILocalVariableAttr DebugImporter::translateImpl(llvm::DILocalVariable *node) {
  // Return nullptr if the scope or type is a cyclic dependency.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  return DILocalVariableAttr::get(
      context, scope, getStringAttrOrNull(node->getRawName()),
      translate(node->getFile()), node->getLine(), node->getArg(),
      node->getAlignInBits(), translate(node->getType()));
}

DIScopeAttr DebugImporter::translateImpl(llvm::DIScope *node) {
  return cast<DIScopeAttr>(translate(static_cast<llvm::DINode *>(node)));
}

DIModuleAttr DebugImporter::translateImpl(llvm::DIModule *node) {
  return DIModuleAttr::get(
      context, translate(node->getFile()), translate(node->getScope()),
      getStringAttrOrNull(node->getRawName()),
      getStringAttrOrNull(node->getRawConfigurationMacros()),
      getStringAttrOrNull(node->getRawIncludePath()),
      getStringAttrOrNull(node->getRawAPINotesFile()), node->getLineNo(),
      node->getIsDecl());
}

DINamespaceAttr DebugImporter::translateImpl(llvm::DINamespace *node) {
  return DINamespaceAttr::get(context, getStringAttrOrNull(node->getRawName()),
                              translate(node->getScope()),
                              node->getExportSymbols());
}

DISubprogramAttr DebugImporter::translateImpl(llvm::DISubprogram *node) {
  // Only definitions require a distinct identifier.
  mlir::DistinctAttr id;
  if (node->isDistinct())
    id = DistinctAttr::create(UnitAttr::get(context));
  std::optional<DISubprogramFlags> subprogramFlags =
      symbolizeDISubprogramFlags(node->getSubprogram()->getSPFlags());
  // Return nullptr if the scope or type is a cyclic dependency.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  DISubroutineTypeAttr type = translate(node->getType());
  if (node->getType() && !type)
    return nullptr;
  return DISubprogramAttr::get(context, id, translate(node->getUnit()), scope,
                               getStringAttrOrNull(node->getRawName()),
                               getStringAttrOrNull(node->getRawLinkageName()),
                               translate(node->getFile()), node->getLine(),
                               node->getScopeLine(), subprogramFlags.value(),
                               type);
}

DISubrangeAttr DebugImporter::translateImpl(llvm::DISubrange *node) {
  auto getIntegerAttrOrNull = [&](llvm::DISubrange::BoundType data) {
    if (auto *constInt = llvm::dyn_cast_or_null<llvm::ConstantInt *>(data))
      return IntegerAttr::get(IntegerType::get(context, 64),
                              constInt->getSExtValue());
    return IntegerAttr();
  };
  IntegerAttr count = getIntegerAttrOrNull(node->getCount());
  IntegerAttr upperBound = getIntegerAttrOrNull(node->getUpperBound());
  // Either count or the upper bound needs to be present. Otherwise, the
  // metadata is invalid. The conversion might fail due to unsupported DI nodes.
  if (!count && !upperBound)
    return {};
  return DISubrangeAttr::get(
      context, count, getIntegerAttrOrNull(node->getLowerBound()), upperBound,
      getIntegerAttrOrNull(node->getStride()));
}

DISubroutineTypeAttr
DebugImporter::translateImpl(llvm::DISubroutineType *node) {
  SmallVector<DITypeAttr> types;
  for (llvm::DIType *type : node->getTypeArray()) {
    if (!type) {
      // A nullptr entry may appear at the beginning or the end of the
      // subroutine types list modeling either a void result type or the type of
      // a variadic argument. Translate the nullptr to an explicit
      // DINullTypeAttr since the attribute list cannot contain a nullptr entry.
      types.push_back(DINullTypeAttr::get(context));
      continue;
    }
    types.push_back(translate(type));
  }
  // Return nullptr if any of the types is a cyclic dependency.
  if (llvm::is_contained(types, nullptr))
    return nullptr;
  return DISubroutineTypeAttr::get(context, node->getCC(), types);
}

DITypeAttr DebugImporter::translateImpl(llvm::DIType *node) {
  return cast<DITypeAttr>(translate(static_cast<llvm::DINode *>(node)));
}

DINodeAttr DebugImporter::translate(llvm::DINode *node) {
  if (!node)
    return nullptr;

  // Check for a cached instance.
  if (DINodeAttr attr = nodeToAttr.lookup(node))
    return attr;

  // Return nullptr if a cyclic dependency is detected since the same node is
  // being traversed twice. This check avoids infinite recursion if the debug
  // metadata contains cycles.
  if (!translationStack.insert(node))
    return nullptr;
  auto guard = llvm::make_scope_exit([&]() { translationStack.pop_back(); });

  // Convert the debug metadata if possible.
  auto translateNode = [this](llvm::DINode *node) -> DINodeAttr {
    if (auto *casted = dyn_cast<llvm::DIBasicType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompileUnit>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompositeType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIDerivedType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIFile>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIGlobalVariable>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILabel>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILexicalBlock>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILexicalBlockFile>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DILocalVariable>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIModule>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DINamespace>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubprogram>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubrange>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubroutineType>(node))
      return translateImpl(casted);
    return nullptr;
  };
  if (DINodeAttr attr = translateNode(node)) {
    nodeToAttr.insert({node, attr});
    return attr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Locations
//===----------------------------------------------------------------------===//

Location DebugImporter::translateLoc(llvm::DILocation *loc) {
  if (!loc)
    return UnknownLoc::get(context);

  // Get the file location of the instruction.
  Location result = FileLineColLoc::get(context, loc->getFilename(),
                                        loc->getLine(), loc->getColumn());

  // Add scope information.
  assert(loc->getScope() && "expected non-null scope");
  result = FusedLocWith<DIScopeAttr>::get({result}, translate(loc->getScope()),
                                          context);

  // Add call site information, if available.
  if (llvm::DILocation *inlinedAt = loc->getInlinedAt())
    result = CallSiteLoc::get(result, translateLoc(inlinedAt));

  return result;
}

DIExpressionAttr DebugImporter::translateExpression(llvm::DIExpression *node) {
  SmallVector<DIExpressionElemAttr> ops;

  // Begin processing the operations.
  for (const llvm::DIExpression::ExprOperand &op : node->expr_ops()) {
    SmallVector<uint64_t> operands;
    operands.reserve(op.getNumArgs());
    for (const auto &i : llvm::seq(op.getNumArgs()))
      operands.push_back(op.getArg(i));
    const auto attr = DIExpressionElemAttr::get(context, op.getOp(), operands);
    ops.push_back(attr);
  }
  return DIExpressionAttr::get(context, ops);
}

DIGlobalVariableExpressionAttr DebugImporter::translateGlobalVariableExpression(
    llvm::DIGlobalVariableExpression *node) {
  return DIGlobalVariableExpressionAttr::get(
      context, translate(node->getVariable()),
      translateExpression(node->getExpression()));
}

StringAttr DebugImporter::getStringAttrOrNull(llvm::MDString *stringNode) {
  if (!stringNode)
    return StringAttr();
  return StringAttr::get(context, stringNode->getString());
}
