//===- LoopAnnotationImporter.cpp - Loop annotation import ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopAnnotationImporter.h"
#include "llvm/IR/Constants.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

namespace {
/// Helper class that keeps the state of one metadata to attribute conversion.
struct LoopMetadataConversion {
  LoopMetadataConversion(const llvm::MDNode *node, ModuleImport &moduleImport,
                         Location loc,
                         LoopAnnotationImporter &loopAnnotationImporter)
      : node(node), moduleImport(moduleImport), loc(loc),
        loopAnnotationImporter(loopAnnotationImporter),
        ctx(loc->getContext()){};
  /// Converts this structs loop metadata node into a LoopAnnotationAttr.
  LoopAnnotationAttr convert();

  LogicalResult initPropertyMap();

  /// Helper function to get and erase a property.
  const llvm::MDNode *lookupAndEraseProperty(StringRef name);

  /// Helper functions to lookup and convert MDNodes into a specifc attribute
  /// kind. These functions return null-attributes if there is no node with the
  /// specified name, or failure, if the node is ill-formatted.
  FailureOr<BoolAttr> lookupUnitNode(StringRef name);
  FailureOr<BoolAttr> lookupBoolNode(StringRef name, bool negated = false);
  FailureOr<IntegerAttr> lookupIntNode(StringRef name);
  FailureOr<llvm::MDNode *> lookupMDNode(StringRef name);
  FailureOr<SmallVector<llvm::MDNode *>> lookupMDNodes(StringRef name);
  FailureOr<LoopAnnotationAttr> lookupFollowupNode(StringRef name);
  FailureOr<BoolAttr> lookupBooleanUnitNode(StringRef enableName,
                                            StringRef disableName,
                                            bool negated = false);

  /// Conversion functions for sub-attributes.
  FailureOr<LoopVectorizeAttr> convertVectorizeAttr();
  FailureOr<LoopInterleaveAttr> convertInterleaveAttr();
  FailureOr<LoopUnrollAttr> convertUnrollAttr();
  FailureOr<LoopUnrollAndJamAttr> convertUnrollAndJamAttr();
  FailureOr<LoopLICMAttr> convertLICMAttr();
  FailureOr<LoopDistributeAttr> convertDistributeAttr();
  FailureOr<LoopPipelineAttr> convertPipelineAttr();
  FailureOr<SmallVector<SymbolRefAttr>> convertParallelAccesses();

  llvm::StringMap<const llvm::MDNode *> propertyMap;
  const llvm::MDNode *node;
  ModuleImport &moduleImport;
  Location loc;
  LoopAnnotationImporter &loopAnnotationImporter;
  MLIRContext *ctx;
};
} // namespace

LogicalResult LoopMetadataConversion::initPropertyMap() {
  // Check if it's a valid node.
  if (node->getNumOperands() == 0 ||
      dyn_cast<llvm::MDNode>(node->getOperand(0)) != node)
    return emitWarning(loc) << "invalid loop node";

  for (const llvm::MDOperand &operand : llvm::drop_begin(node->operands())) {
    // Skip over DILocations.
    if (isa<llvm::DILocation>(operand))
      continue;

    auto *property = dyn_cast<llvm::MDNode>(operand);
    if (!property)
      return emitWarning(loc) << "expected all loop properties to be either "
                                 "debug locations or metadata nodes";

    if (property->getNumOperands() == 0)
      return emitWarning(loc) << "cannot import empty loop property";

    auto *nameNode = dyn_cast<llvm::MDString>(property->getOperand(0));
    if (!nameNode)
      return emitWarning(loc) << "cannot import loop property without a name";
    StringRef name = nameNode->getString();

    bool succ = propertyMap.try_emplace(name, property).second;
    if (!succ)
      return emitWarning(loc)
             << "cannot import loop properties with duplicated names " << name;
  }

  return success();
}

const llvm::MDNode *
LoopMetadataConversion::lookupAndEraseProperty(StringRef name) {
  auto it = propertyMap.find(name);
  if (it == propertyMap.end())
    return nullptr;
  const llvm::MDNode *property = it->getValue();
  propertyMap.erase(it);
  return property;
}

FailureOr<BoolAttr> LoopMetadataConversion::lookupUnitNode(StringRef name) {
  const llvm::MDNode *property = lookupAndEraseProperty(name);
  if (!property)
    return BoolAttr(nullptr);

  if (property->getNumOperands() != 1)
    return emitWarning(loc)
           << "expected metadata node " << name << " to hold no value";

  return BoolAttr::get(ctx, true);
}

FailureOr<BoolAttr> LoopMetadataConversion::lookupBooleanUnitNode(
    StringRef enableName, StringRef disableName, bool negated) {
  auto enable = lookupUnitNode(enableName);
  auto disable = lookupUnitNode(disableName);
  if (failed(enable) || failed(disable))
    return failure();

  if (*enable && *disable)
    return emitWarning(loc)
           << "expected metadata nodes " << enableName << " and " << disableName
           << " to be mutually exclusive.";

  if (*enable)
    return BoolAttr::get(ctx, !negated);

  if (*disable)
    return BoolAttr::get(ctx, negated);
  return BoolAttr(nullptr);
}

FailureOr<BoolAttr> LoopMetadataConversion::lookupBoolNode(StringRef name,
                                                           bool negated) {
  const llvm::MDNode *property = lookupAndEraseProperty(name);
  if (!property)
    return BoolAttr(nullptr);

  auto emitNodeWarning = [&]() {
    return emitWarning(loc)
           << "expected metadata node " << name << " to hold a boolean value";
  };

  if (property->getNumOperands() != 2)
    return emitNodeWarning();
  llvm::ConstantInt *val =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(property->getOperand(1));
  if (!val || val->getBitWidth() != 1)
    return emitNodeWarning();

  return BoolAttr::get(ctx, val->getValue().getLimitedValue(1) ^ negated);
}

FailureOr<IntegerAttr> LoopMetadataConversion::lookupIntNode(StringRef name) {
  const llvm::MDNode *property = lookupAndEraseProperty(name);
  if (!property)
    return IntegerAttr(nullptr);

  auto emitNodeWarning = [&]() {
    return emitWarning(loc)
           << "expected metadata node " << name << " to hold an i32 value";
  };

  if (property->getNumOperands() != 2)
    return emitNodeWarning();

  llvm::ConstantInt *val =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(property->getOperand(1));
  if (!val || val->getBitWidth() != 32)
    return emitNodeWarning();

  return IntegerAttr::get(IntegerType::get(ctx, 32),
                          val->getValue().getLimitedValue());
}

FailureOr<llvm::MDNode *> LoopMetadataConversion::lookupMDNode(StringRef name) {
  const llvm::MDNode *property = lookupAndEraseProperty(name);
  if (!property)
    return nullptr;

  auto emitNodeWarning = [&]() {
    return emitWarning(loc)
           << "expected metadata node " << name << " to hold an MDNode";
  };

  if (property->getNumOperands() != 2)
    return emitNodeWarning();

  auto *node = dyn_cast<llvm::MDNode>(property->getOperand(1));
  if (!node)
    return emitNodeWarning();

  return node;
}

FailureOr<SmallVector<llvm::MDNode *>>
LoopMetadataConversion::lookupMDNodes(StringRef name) {
  const llvm::MDNode *property = lookupAndEraseProperty(name);
  SmallVector<llvm::MDNode *> res;
  if (!property)
    return res;

  auto emitNodeWarning = [&]() {
    return emitWarning(loc) << "expected metadata node " << name
                            << " to hold one or multiple MDNodes";
  };

  if (property->getNumOperands() < 2)
    return emitNodeWarning();

  for (unsigned i = 1, e = property->getNumOperands(); i < e; ++i) {
    auto *node = dyn_cast<llvm::MDNode>(property->getOperand(i));
    if (!node)
      return emitNodeWarning();
    res.push_back(node);
  }

  return res;
}

FailureOr<LoopAnnotationAttr>
LoopMetadataConversion::lookupFollowupNode(StringRef name) {
  auto node = lookupMDNode(name);
  if (failed(node))
    return failure();
  if (*node == nullptr)
    return LoopAnnotationAttr(nullptr);

  return loopAnnotationImporter.translate(*node, loc);
}

static bool isEmptyOrNull(const Attribute attr) { return !attr; }

template <typename T>
static bool isEmptyOrNull(const SmallVectorImpl<T> &vec) {
  return vec.empty();
}

/// Helper function that only creates and attribute of type T if all argument
/// conversion were successfull and at least one of them holds a non-null value.
template <typename T, typename... P>
static T createIfNonNull(MLIRContext *ctx, const P &...args) {
  bool anyFailed = (failed(args) || ...);
  if (anyFailed)
    return {};

  bool allEmpty = (isEmptyOrNull(*args) && ...);
  if (allEmpty)
    return {};

  return T::get(ctx, *args...);
}

FailureOr<LoopVectorizeAttr> LoopMetadataConversion::convertVectorizeAttr() {
  FailureOr<BoolAttr> enable =
      lookupBoolNode("llvm.loop.vectorize.enable", true);
  FailureOr<BoolAttr> predicateEnable =
      lookupBoolNode("llvm.loop.vectorize.predicate.enable");
  FailureOr<BoolAttr> scalableEnable =
      lookupBoolNode("llvm.loop.vectorize.scalable.enable");
  FailureOr<IntegerAttr> width = lookupIntNode("llvm.loop.vectorize.width");
  FailureOr<LoopAnnotationAttr> followupVec =
      lookupFollowupNode("llvm.loop.vectorize.followup_vectorized");
  FailureOr<LoopAnnotationAttr> followupEpi =
      lookupFollowupNode("llvm.loop.vectorize.followup_epilogue");
  FailureOr<LoopAnnotationAttr> followupAll =
      lookupFollowupNode("llvm.loop.vectorize.followup_all");

  return createIfNonNull<LoopVectorizeAttr>(ctx, enable, predicateEnable,
                                            scalableEnable, width, followupVec,
                                            followupEpi, followupAll);
}

FailureOr<LoopInterleaveAttr> LoopMetadataConversion::convertInterleaveAttr() {
  FailureOr<IntegerAttr> count = lookupIntNode("llvm.loop.interleave.count");
  return createIfNonNull<LoopInterleaveAttr>(ctx, count);
}

FailureOr<LoopUnrollAttr> LoopMetadataConversion::convertUnrollAttr() {
  FailureOr<BoolAttr> disable = lookupBooleanUnitNode(
      "llvm.loop.unroll.enable", "llvm.loop.unroll.disable", /*negated=*/true);
  FailureOr<IntegerAttr> count = lookupIntNode("llvm.loop.unroll.count");
  FailureOr<BoolAttr> runtimeDisable =
      lookupUnitNode("llvm.loop.unroll.runtime.disable");
  FailureOr<BoolAttr> full = lookupUnitNode("llvm.loop.unroll.full");
  FailureOr<LoopAnnotationAttr> followup =
      lookupFollowupNode("llvm.loop.unroll.followup");
  FailureOr<LoopAnnotationAttr> followupRemainder =
      lookupFollowupNode("llvm.loop.unroll.followup_remainder");

  return createIfNonNull<LoopUnrollAttr>(ctx, disable, count, runtimeDisable,
                                         full, followup, followupRemainder);
}

FailureOr<LoopUnrollAndJamAttr>
LoopMetadataConversion::convertUnrollAndJamAttr() {
  FailureOr<BoolAttr> disable = lookupBooleanUnitNode(
      "llvm.loop.unroll_and_jam.enable", "llvm.loop.unroll_and_jam.disable",
      /*negated=*/true);
  FailureOr<IntegerAttr> count =
      lookupIntNode("llvm.loop.unroll_and_jam.count");
  FailureOr<LoopAnnotationAttr> followupOuter =
      lookupFollowupNode("llvm.loop.unroll_and_jam.followup_outer");
  FailureOr<LoopAnnotationAttr> followupInner =
      lookupFollowupNode("llvm.loop.unroll_and_jam.followup_inner");
  FailureOr<LoopAnnotationAttr> followupRemainderOuter =
      lookupFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_outer");
  FailureOr<LoopAnnotationAttr> followupRemainderInner =
      lookupFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_inner");
  FailureOr<LoopAnnotationAttr> followupAll =
      lookupFollowupNode("llvm.loop.unroll_and_jam.followup_all");
  return createIfNonNull<LoopUnrollAndJamAttr>(
      ctx, disable, count, followupOuter, followupInner, followupRemainderOuter,
      followupRemainderInner, followupAll);
}

FailureOr<LoopLICMAttr> LoopMetadataConversion::convertLICMAttr() {
  FailureOr<BoolAttr> disable = lookupUnitNode("llvm.licm.disable");
  FailureOr<BoolAttr> versioningDisable =
      lookupUnitNode("llvm.loop.licm_versioning.disable");
  return createIfNonNull<LoopLICMAttr>(ctx, disable, versioningDisable);
}

FailureOr<LoopDistributeAttr> LoopMetadataConversion::convertDistributeAttr() {
  FailureOr<BoolAttr> disable =
      lookupBoolNode("llvm.loop.distribute.enable", true);
  FailureOr<LoopAnnotationAttr> followupCoincident =
      lookupFollowupNode("llvm.loop.distribute.followup_coincident");
  FailureOr<LoopAnnotationAttr> followupSequential =
      lookupFollowupNode("llvm.loop.distribute.followup_sequential");
  FailureOr<LoopAnnotationAttr> followupFallback =
      lookupFollowupNode("llvm.loop.distribute.followup_fallback");
  FailureOr<LoopAnnotationAttr> followupAll =
      lookupFollowupNode("llvm.loop.distribute.followup_all");
  return createIfNonNull<LoopDistributeAttr>(ctx, disable, followupCoincident,
                                             followupSequential,
                                             followupFallback, followupAll);
}

FailureOr<LoopPipelineAttr> LoopMetadataConversion::convertPipelineAttr() {
  FailureOr<BoolAttr> disable = lookupBoolNode("llvm.loop.pipeline.disable");
  FailureOr<IntegerAttr> initiationinterval =
      lookupIntNode("llvm.loop.pipeline.initiationinterval");
  return createIfNonNull<LoopPipelineAttr>(ctx, disable, initiationinterval);
}

FailureOr<SmallVector<SymbolRefAttr>>
LoopMetadataConversion::convertParallelAccesses() {
  FailureOr<SmallVector<llvm::MDNode *>> nodes =
      lookupMDNodes("llvm.loop.parallel_accesses");
  if (failed(nodes))
    return failure();
  SmallVector<SymbolRefAttr> refs;
  for (llvm::MDNode *node : *nodes) {
    FailureOr<SmallVector<SymbolRefAttr>> accessGroups =
        moduleImport.lookupAccessGroupAttrs(node);
    if (failed(accessGroups))
      return emitWarning(loc) << "could not lookup access group";
    llvm::append_range(refs, *accessGroups);
  }
  return refs;
}

LoopAnnotationAttr LoopMetadataConversion::convert() {
  if (failed(initPropertyMap()))
    return {};

  FailureOr<BoolAttr> disableNonForced =
      lookupUnitNode("llvm.loop.disable_nonforced");
  FailureOr<LoopVectorizeAttr> vecAttr = convertVectorizeAttr();
  FailureOr<LoopInterleaveAttr> interleaveAttr = convertInterleaveAttr();
  FailureOr<LoopUnrollAttr> unrollAttr = convertUnrollAttr();
  FailureOr<LoopUnrollAndJamAttr> unrollAndJamAttr = convertUnrollAndJamAttr();
  FailureOr<LoopLICMAttr> licmAttr = convertLICMAttr();
  FailureOr<LoopDistributeAttr> distributeAttr = convertDistributeAttr();
  FailureOr<LoopPipelineAttr> pipelineAttr = convertPipelineAttr();
  FailureOr<BoolAttr> mustProgress = lookupUnitNode("llvm.loop.mustprogress");
  FailureOr<SmallVector<SymbolRefAttr>> parallelAccesses =
      convertParallelAccesses();

  // Drop the metadata if there are parts that cannot be imported.
  if (!propertyMap.empty()) {
    for (auto name : propertyMap.keys())
      emitWarning(loc) << "unknown loop annotation " << name;
    return {};
  }

  return createIfNonNull<LoopAnnotationAttr>(
      ctx, disableNonForced, vecAttr, interleaveAttr, unrollAttr,
      unrollAndJamAttr, licmAttr, distributeAttr, pipelineAttr, mustProgress,
      parallelAccesses);
}

LoopAnnotationAttr LoopAnnotationImporter::translate(const llvm::MDNode *node,
                                                     Location loc) {
  if (!node)
    return {};

  // Note: This check is necessary to distinguish between failed translations
  // and not yet attempted translations.
  auto it = loopMetadataMapping.find(node);
  if (it != loopMetadataMapping.end())
    return it->getSecond();

  LoopAnnotationAttr attr =
      LoopMetadataConversion(node, moduleImport, loc, *this).convert();

  mapLoopMetadata(node, attr);
  return attr;
}
