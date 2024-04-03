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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

DebugImporter::DebugImporter(ModuleOp mlirModule)
    : recursionPruner(mlirModule.getContext()),
      context(mlirModule.getContext()), mlirModule(mlirModule) {}

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
      context, getOrCreateDistinctID(node), node->getSourceLanguage(),
      translate(node->getFile()), getStringAttrOrNull(node->getRawProducer()),
      node->isOptimized(), emissionKind.value());
}

DICompositeTypeAttr DebugImporter::translateImpl(llvm::DICompositeType *node) {
  std::optional<DIFlags> flags = symbolizeDIFlags(node->getFlags());
  SmallVector<DINodeAttr> elements;
  for (llvm::DINode *element : node->getElements()) {
    assert(element && "expected a non-null element type");
    elements.push_back(translate(element));
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
      context, node->getTag(), /*recId=*/{},
      getStringAttrOrNull(node->getRawName()), translate(node->getFile()),
      node->getLine(), translate(node->getScope()), baseType,
      flags.value_or(DIFlags::Zero), node->getSizeInBits(),
      node->getAlignInBits(), elements);
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
      node->getOffsetInBits(), extraData);
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
  return DISubprogramAttr::get(context, id, translate(node->getUnit()), scope,
                               getStringAttrOrNull(node->getRawName()),
                               getStringAttrOrNull(node->getRawLinkageName()),
                               translate(node->getFile()), node->getLine(),
                               node->getScopeLine(), *subprogramFlags, type);
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
  if (DINodeAttr attr = nodeToAttr.lookup(node))
    return attr;

  // Register with the recursive translator. If it can be handled without
  // recursing into it, return the result immediately.
  if (DINodeAttr attr = recursionPruner.pruneOrPushTranslationStack(node))
    return attr;

  auto guard = llvm::make_scope_exit(
      [&]() { recursionPruner.popTranslationStack(node); });

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
    auto [result, isSelfContained] =
        recursionPruner.finalizeTranslation(node, attr);
    // Only cache fully self-contained nodes.
    if (isSelfContained)
      nodeToAttr.try_emplace(node, result);
    return result;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// RecursionPruner
//===----------------------------------------------------------------------===//
/// Get the `getRecSelf` constructor for the translated type of `node` if its
/// translated DITypeAttr supports recursion. Otherwise, returns nullptr.
static function_ref<DIRecursiveTypeAttrInterface(DistinctAttr)>
getRecSelfConstructor(llvm::DINode *node) {
  using CtorType = function_ref<DIRecursiveTypeAttrInterface(DistinctAttr)>;
  return TypeSwitch<llvm::DINode *, CtorType>(node)
      .Case([&](llvm::DICompositeType *) {
        return CtorType(DICompositeTypeAttr::getRecSelf);
      })
      .Default(CtorType());
}

/// An attribute replacer that replaces nested recursive decls with recursive
/// self-references instead.
///
/// - Recurses down the attribute tree while replacing attributes based on the
///   provided replacement map.
/// - Keeps track of the currently open recursive declarations, and upon
///   encountering a duplicate declaration, replace with a self-ref instead.
static Attribute replaceAndPruneRecursiveTypesImpl(
    Attribute node,
    const DenseMap<DIRecursiveTypeAttrInterface, DINodeAttr> &mapping,
    DenseSet<DistinctAttr> &openDecls) {
  DistinctAttr recId;
  if (auto recType = dyn_cast<DIRecursiveTypeAttrInterface>(node)) {
    recId = recType.getRecId();

    // Configure context.
    if (recId) {
      if (recType.isRecSelf()) {
        // Replace selfRef based on the provided mapping.
        if (DINodeAttr replacement = mapping.lookup(recType))
          return replaceAndPruneRecursiveTypesImpl(replacement, mapping,
                                                   openDecls);
        return node;
      }

      auto [_, inserted] = openDecls.insert(recId);
      if (!inserted) {
        // This is a nested decl. Replace with recSelf.
        return recType.getRecSelf(recId);
      }
    }
  }

  // Collect sub attrs.
  SmallVector<Attribute> attrs;
  SmallVector<Type> types;
  node.walkImmediateSubElements(
      [&attrs](Attribute attr) { attrs.push_back(attr); },
      [&types](Type type) { types.push_back(type); });

  // Recurse into attributes.
  bool changed = false;
  for (auto it = attrs.begin(); it != attrs.end(); it++) {
    Attribute replaced =
        replaceAndPruneRecursiveTypesImpl(*it, mapping, openDecls);
    if (replaced != *it) {
      *it = replaced;
      changed = true;
    }
  }

  Attribute result = node;
  if (changed)
    result = result.replaceImmediateSubElements(attrs, types);

  // Reset context.
  if (recId)
    openDecls.erase(recId);

  return result;
}

static Attribute replaceAndPruneRecursiveTypes(
    DINodeAttr node,
    const DenseMap<DIRecursiveTypeAttrInterface, DINodeAttr> &mapping) {
  DenseSet<DistinctAttr> openDecls;
  return replaceAndPruneRecursiveTypesImpl(node, mapping, openDecls);
}

DINodeAttr DebugImporter::RecursionPruner::pruneOrPushTranslationStack(
    llvm::DINode *node) {
  // Lookup the cache first.
  auto [result, unboundSelfRefs] = lookup(node);
  if (result) {
    // Need to inject unbound self-refs into the previous layer.
    if (!unboundSelfRefs.empty())
      translationStack.back().second.unboundSelfRefs.insert(
          unboundSelfRefs.begin(), unboundSelfRefs.end());
    return result;
  }

  // If the node type is capable of being recursive, check if it's seen
  // before.
  auto recSelfCtor = getRecSelfConstructor(node);
  if (recSelfCtor) {
    // If a cyclic dependency is detected since the same node is being
    // traversed twice, emit a recursive self type, and mark the duplicate
    // node on the translationStack so it can emit a recursive decl type.
    auto [iter, inserted] = translationStack.try_emplace(node);
    if (!inserted) {
      // The original node may have already been assigned a recursive ID from
      // a different self-reference. Use that if possible.
      DIRecursiveTypeAttrInterface recSelf = iter->second.recSelf;
      if (!recSelf) {
        DistinctAttr recId = DistinctAttr::create(UnitAttr::get(context));
        recSelf = recSelfCtor(recId);
        iter->second.recSelf = recSelf;
      }
      // Inject the self-ref into the previous layer.
      translationStack.back().second.unboundSelfRefs.insert(recSelf);
      return cast<DINodeAttr>(recSelf);
    }
  }
  return nullptr;
}

std::pair<DINodeAttr, bool>
DebugImporter::RecursionPruner::finalizeTranslation(llvm::DINode *node,
                                                    DINodeAttr result) {
  // If `node` is not a potentially recursive type, it will not be on the
  // translation stack. Nothing to set in this case.
  if (translationStack.empty())
    return {result, true};
  if (translationStack.back().first != node)
    return {result, translationStack.back().second.unboundSelfRefs.empty()};

  TranslationState &state = translationStack.back().second;

  // If this node is actually recursive, set the recId onto `result`.
  if (DIRecursiveTypeAttrInterface recSelf = state.recSelf) {
    auto recType = cast<DIRecursiveTypeAttrInterface>(result);
    result = cast<DINodeAttr>(recType.withRecId(recSelf.getRecId()));

    // Remove this recSelf from the set of unbound selfRefs.
    state.unboundSelfRefs.erase(recSelf);

    // Insert the newly resolved recursive type into the cache entries that
    // rely on it.
    // Only need to look at the caches at this level.
    uint64_t numRemaining = state.cacheSize;
    for (CachedTranslation &cacheEntry :
         llvm::make_second_range(llvm::reverse(cache))) {
      if (numRemaining == 0)
        break;
      --numRemaining;

      if (auto refIter = cacheEntry.pendingReplacements.find(recSelf);
          refIter != cacheEntry.pendingReplacements.end())
        refIter->second = result;
    }
  }

  // Insert the current result into the cache.
  state.cacheSize++;
  auto [iter, inserted] = cache.try_emplace(node);
  assert(inserted && "invalid state: caching the same DINode twice");
  iter->second.attr = result;

  // If this node had any unbound self-refs free when it is registered into
  // the cache, set up replacement placeholders: This result will need these
  // unbound self-refs to be replaced before being used.
  for (DIRecursiveTypeAttrInterface selfRef : state.unboundSelfRefs)
    iter->second.pendingReplacements.try_emplace(selfRef, nullptr);

  return {result, state.unboundSelfRefs.empty()};
}

void DebugImporter::RecursionPruner::popTranslationStack(llvm::DINode *node) {
  // If `node` is not a potentially recursive type, it will not be on the
  // translation stack. Nothing to handle in this case.
  if (translationStack.empty() || translationStack.back().first != node)
    return;

  // At the end of the stack, all unbound self-refs must be resolved already,
  // and the entire cache should be accounted for.
  TranslationState &currLayerState = translationStack.back().second;
  if (translationStack.size() == 1) {
    assert(currLayerState.unboundSelfRefs.empty() &&
           "internal error: unbound recursive self reference at top level.");
    assert(currLayerState.cacheSize == cache.size() &&
           "internal error: inconsistent cache size");
    translationStack.pop_back();
    cache.clear();
    return;
  }

  // Copy unboundSelfRefs down to the previous level.
  TranslationState &nextLayerState = (++translationStack.rbegin())->second;
  nextLayerState.unboundSelfRefs.insert(currLayerState.unboundSelfRefs.begin(),
                                        currLayerState.unboundSelfRefs.end());

  // The current layer cache is now considered part of the lower layer cache.
  nextLayerState.cacheSize += currLayerState.cacheSize;

  // Finally pop off this layer when all bookkeeping is done.
  translationStack.pop_back();
}

std::pair<DINodeAttr, DenseSet<DIRecursiveTypeAttrInterface>>
DebugImporter::RecursionPruner::lookup(llvm::DINode *node) {
  auto cacheIter = cache.find(node);
  if (cacheIter == cache.end())
    return {};

  CachedTranslation &entry = cacheIter->second;

  if (entry.pendingReplacements.empty())
    return std::make_pair(entry.attr, DenseSet<DIRecursiveTypeAttrInterface>{});

  Attribute replacedAttr =
      replaceAndPruneRecursiveTypes(entry.attr, entry.pendingReplacements);
  DINodeAttr result = cast<DINodeAttr>(replacedAttr);

  // Update cache entry to save replaced version and remove already-applied
  // replacements.
  entry.attr = result;
  DenseSet<DIRecursiveTypeAttrInterface> unboundRefs;
  DenseSet<DIRecursiveTypeAttrInterface> boundRefs;
  for (auto [refSelf, replacement] : entry.pendingReplacements) {
    if (replacement)
      boundRefs.insert(refSelf);
    else
      unboundRefs.insert(refSelf);
  }

  for (DIRecursiveTypeAttrInterface ref : boundRefs)
    entry.pendingReplacements.erase(ref);

  return std::make_pair(result, unboundRefs);
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

DistinctAttr DebugImporter::getOrCreateDistinctID(llvm::DINode *node) {
  DistinctAttr &id = nodeToDistinctAttr[node];
  if (!id)
    id = DistinctAttr::create(UnitAttr::get(context));
  return id;
}
