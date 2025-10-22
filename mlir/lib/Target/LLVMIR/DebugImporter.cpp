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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

DebugImporter::DebugImporter(ModuleOp mlirModule,
                             bool dropDICompositeTypeElements)
    : cache([&](llvm::DINode *node) { return createRecSelf(node); }),
      context(mlirModule.getContext()), mlirModule(mlirModule),
      dropDICompositeTypeElements(dropDICompositeTypeElements) {}

Location DebugImporter::translateFuncLocation(llvm::Function *func) {
  llvm::DISubprogram *subprogram = func->getSubprogram();
  if (!subprogram)
    return UnknownLoc::get(context);

  // Add a fused location to link the subprogram information.
  StringAttr fileName = StringAttr::get(context, subprogram->getFilename());
  return FusedLocWith<DISubprogramAttr>::get(
      {FileLineColLoc::get(fileName, subprogram->getLine(), /*column=*/0)},
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
  std::optional<DINameTableKind> nameTableKind = symbolizeDINameTableKind(
      static_cast<
          std::underlying_type_t<llvm::DICompileUnit::DebugNameTableKind>>(
          node->getNameTableKind()));
  return DICompileUnitAttr::get(
      context, getOrCreateDistinctID(node),
      node->getSourceLanguage().getUnversionedName(),
      translate(node->getFile()), getStringAttrOrNull(node->getRawProducer()),
      node->isOptimized(), emissionKind.value(), nameTableKind.value(),
      getStringAttrOrNull(node->getRawSplitDebugFilename()));
}

DICompositeTypeAttr DebugImporter::translateImpl(llvm::DICompositeType *node) {
  std::optional<DIFlags> flags = symbolizeDIFlags(node->getFlags());
  SmallVector<DINodeAttr> elements;

  // A vector always requires an element.
  bool isVectorType = flags && bitEnumContainsAll(*flags, DIFlags::Vector);
  if (isVectorType || !dropDICompositeTypeElements) {
    for (llvm::DINode *element : node->getElements()) {
      assert(element && "expected a non-null element type");
      elements.push_back(translate(element));
    }
  }
  // Drop the elements parameter if any of the elements are invalid.
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
      node->getAlignInBits(), translateExpression(node->getDataLocationExp()),
      translateExpression(node->getRankExp()),
      translateExpression(node->getAllocatedExp()),
      translateExpression(node->getAssociatedExp()), elements);
}

DIDerivedTypeAttr DebugImporter::translateImpl(llvm::DIDerivedType *node) {
  // Return nullptr if the base type is invalid.
  DITypeAttr baseType = translate(node->getBaseType());
  if (node->getBaseType() && !baseType)
    return nullptr;
  DINodeAttr extraData =
      translate(dyn_cast_or_null<llvm::DINode>(node->getExtraData()));
  return DIDerivedTypeAttr::get(
      context, node->getTag(), getStringAttrOrNull(node->getRawName()),
      baseType, node->getSizeInBits(), node->getAlignInBits(),
      node->getOffsetInBits(), node->getDWARFAddressSpace(), extraData);
}

DIStringTypeAttr DebugImporter::translateImpl(llvm::DIStringType *node) {
  return DIStringTypeAttr::get(
      context, node->getTag(), getStringAttrOrNull(node->getRawName()),
      node->getSizeInBits(), node->getAlignInBits(),
      translate(node->getStringLength()),
      translateExpression(node->getStringLengthExp()),
      translateExpression(node->getStringLocationExp()), node->getEncoding());
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
      node->getAlignInBits(), translate(node->getType()),
      symbolizeDIFlags(node->getFlags()).value_or(DIFlags::Zero));
}

DIVariableAttr DebugImporter::translateImpl(llvm::DIVariable *node) {
  return cast<DIVariableAttr>(translate(static_cast<llvm::DINode *>(node)));
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

DIImportedEntityAttr
DebugImporter::translateImpl(llvm::DIImportedEntity *node) {
  SmallVector<DINodeAttr> elements;
  for (llvm::DINode *element : node->getElements()) {
    assert(element && "expected a non-null element type");
    elements.push_back(translate(element));
  }

  return DIImportedEntityAttr::get(
      context, node->getTag(), translate(node->getScope()),
      translate(node->getEntity()), translate(node->getFile()), node->getLine(),
      getStringAttrOrNull(node->getRawName()), elements);
}

DISubprogramAttr DebugImporter::translateImpl(llvm::DISubprogram *node) {
  // Only definitions require a distinct identifier.
  mlir::DistinctAttr id;
  if (node->isDistinct())
    id = getOrCreateDistinctID(node);

  // Return nullptr if the scope or type is invalid.
  DIScopeAttr scope = translate(node->getScope());
  if (node->getScope() && !scope)
    return nullptr;
  std::optional<DISubprogramFlags> subprogramFlags =
      symbolizeDISubprogramFlags(node->getSubprogram()->getSPFlags());
  assert(subprogramFlags && "expected valid subprogram flags");
  DISubroutineTypeAttr type = translate(node->getType());
  if (node->getType() && !type)
    return nullptr;

  // Convert the retained nodes but drop all of them if one of them is invalid.
  SmallVector<DINodeAttr> retainedNodes;
  for (llvm::DINode *retainedNode : node->getRetainedNodes())
    retainedNodes.push_back(translate(retainedNode));
  if (llvm::is_contained(retainedNodes, nullptr))
    retainedNodes.clear();

  SmallVector<DINodeAttr> annotations;
  // We currently only support `string` values for annotations on the MLIR side.
  // Theoretically we could support other primitives, but LLVM is not using
  // other types in practice.
  if (llvm::DINodeArray rawAnns = node->getAnnotations(); rawAnns) {
    for (size_t i = 0, e = rawAnns->getNumOperands(); i < e; ++i) {
      const llvm::MDTuple *tuple = cast<llvm::MDTuple>(rawAnns->getOperand(i));
      if (tuple->getNumOperands() != 2)
        continue;
      const llvm::MDString *name = cast<llvm::MDString>(tuple->getOperand(0));
      const llvm::MDString *value =
          dyn_cast<llvm::MDString>(tuple->getOperand(1));
      if (name && value) {
        annotations.push_back(DIAnnotationAttr::get(
            context, StringAttr::get(context, name->getString()),
            StringAttr::get(context, value->getString())));
      }
    }
  }

  return DISubprogramAttr::get(context, id, translate(node->getUnit()), scope,
                               getStringAttrOrNull(node->getRawName()),
                               getStringAttrOrNull(node->getRawLinkageName()),
                               translate(node->getFile()), node->getLine(),
                               node->getScopeLine(), *subprogramFlags, type,
                               retainedNodes, annotations);
}

DISubrangeAttr DebugImporter::translateImpl(llvm::DISubrange *node) {
  auto getAttrOrNull = [&](llvm::DISubrange::BoundType data) -> Attribute {
    if (data.isNull())
      return nullptr;
    if (auto *constInt = dyn_cast<llvm::ConstantInt *>(data))
      return IntegerAttr::get(IntegerType::get(context, 64),
                              constInt->getSExtValue());
    if (auto *expr = dyn_cast<llvm::DIExpression *>(data))
      return translateExpression(expr);
    if (auto *var = dyn_cast<llvm::DIVariable *>(data)) {
      if (auto *local = dyn_cast<llvm::DILocalVariable>(var))
        return translate(local);
      if (auto *global = dyn_cast<llvm::DIGlobalVariable>(var))
        return translate(global);
      return nullptr;
    }
    return nullptr;
  };
  Attribute count = getAttrOrNull(node->getCount());
  Attribute upperBound = getAttrOrNull(node->getUpperBound());
  // Either count or the upper bound needs to be present. Otherwise, the
  // metadata is invalid. The conversion might fail due to unsupported DI nodes.
  if (!count && !upperBound)
    return {};
  return DISubrangeAttr::get(context, count,
                             getAttrOrNull(node->getLowerBound()), upperBound,
                             getAttrOrNull(node->getStride()));
}

DICommonBlockAttr DebugImporter::translateImpl(llvm::DICommonBlock *node) {
  return DICommonBlockAttr::get(context, translate(node->getScope()),
                                translate(node->getDecl()),
                                getStringAttrOrNull(node->getRawName()),
                                translate(node->getFile()), node->getLineNo());
}

DIGenericSubrangeAttr
DebugImporter::translateImpl(llvm::DIGenericSubrange *node) {
  auto getAttrOrNull =
      [&](llvm::DIGenericSubrange::BoundType data) -> Attribute {
    if (data.isNull())
      return nullptr;
    if (auto *expr = dyn_cast<llvm::DIExpression *>(data))
      return translateExpression(expr);
    if (auto *var = dyn_cast<llvm::DIVariable *>(data)) {
      if (auto *local = dyn_cast<llvm::DILocalVariable>(var))
        return translate(local);
      if (auto *global = dyn_cast<llvm::DIGlobalVariable>(var))
        return translate(global);
      return nullptr;
    }
    return nullptr;
  };
  Attribute count = getAttrOrNull(node->getCount());
  Attribute upperBound = getAttrOrNull(node->getUpperBound());
  Attribute lowerBound = getAttrOrNull(node->getLowerBound());
  Attribute stride = getAttrOrNull(node->getStride());
  // Either count or the upper bound needs to be present. Otherwise, the
  // metadata is invalid.
  if (!count && !upperBound)
    return {};
  return DIGenericSubrangeAttr::get(context, count, lowerBound, upperBound,
                                    stride);
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
  // Return nullptr if any of the types is invalid.
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
  auto cacheEntry = cache.lookupOrInit(node);
  if (std::optional<DINodeAttr> result = cacheEntry.get())
    return *result;

  // Convert the debug metadata if possible.
  auto translateNode = [this](llvm::DINode *node) -> DINodeAttr {
    if (auto *casted = dyn_cast<llvm::DIBasicType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICommonBlock>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompileUnit>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DICompositeType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIDerivedType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIStringType>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIFile>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIGlobalVariable>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DIImportedEntity>(node))
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
    if (auto *casted = dyn_cast<llvm::DIGenericSubrange>(node))
      return translateImpl(casted);
    if (auto *casted = dyn_cast<llvm::DISubroutineType>(node))
      return translateImpl(casted);
    return nullptr;
  };
  if (DINodeAttr attr = translateNode(node)) {
    // If this node was repeated, lookup its recursive ID and assign it to the
    // base result.
    if (cacheEntry.wasRepeated()) {
      DistinctAttr recId = nodeToRecId.lookup(node);
      auto recType = cast<DIRecursiveTypeAttrInterface>(attr);
      attr = cast<DINodeAttr>(recType.withRecId(recId));
    }
    cacheEntry.resolve(attr);
    return attr;
  }
  cacheEntry.resolve(nullptr);
  return nullptr;
}

/// Get the `getRecSelf` constructor for the translated type of `node` if its
/// translated DITypeAttr supports recursion. Otherwise, returns nullptr.
static function_ref<DIRecursiveTypeAttrInterface(DistinctAttr)>
getRecSelfConstructor(llvm::DINode *node) {
  using CtorType = function_ref<DIRecursiveTypeAttrInterface(DistinctAttr)>;
  return TypeSwitch<llvm::DINode *, CtorType>(node)
      .Case([&](llvm::DICompositeType *) {
        return CtorType(DICompositeTypeAttr::getRecSelf);
      })
      .Case([&](llvm::DISubprogram *) {
        return CtorType(DISubprogramAttr::getRecSelf);
      })
      .Default(CtorType());
}

std::optional<DINodeAttr> DebugImporter::createRecSelf(llvm::DINode *node) {
  auto recSelfCtor = getRecSelfConstructor(node);
  if (!recSelfCtor)
    return std::nullopt;

  // The original node may have already been assigned a recursive ID from
  // a different self-reference. Use that if possible.
  DistinctAttr recId = nodeToRecId.lookup(node);
  if (!recId) {
    recId = DistinctAttr::create(UnitAttr::get(context));
    nodeToRecId[node] = recId;
  }
  DIRecursiveTypeAttrInterface recSelf = recSelfCtor(recId);
  return cast<DINodeAttr>(recSelf);
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
  if (!node)
    return nullptr;

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

DistinctAttr DebugImporter::getOrCreateDistinctID(llvm::DINode *node) {
  DistinctAttr &id = nodeToDistinctAttr[node];
  if (!id)
    id = DistinctAttr::create(UnitAttr::get(context));
  return id;
}
