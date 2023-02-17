//===- LoopAnnotationTranslation.cpp - Loop annotation export -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopAnnotationTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

namespace {
/// Helper class that keeps the state of one attribute to metadata conversion.
struct LoopAnnotationConversion {
  LoopAnnotationConversion(LoopAnnotationAttr attr, Operation *op,
                           LoopAnnotationTranslation &loopAnnotationTranslation,
                           llvm::LLVMContext &ctx)
      : attr(attr), op(op),
        loopAnnotationTranslation(loopAnnotationTranslation), ctx(ctx) {}

  /// Converts this struct's loop annotation into a corresponding LLVMIR
  /// metadata representation.
  llvm::MDNode *convert();

  /// Conversion functions for different payload attribute kinds.
  void addUnitNode(StringRef name);
  void addUnitNode(StringRef name, BoolAttr attr);
  void addI32NodeWithVal(StringRef name, uint32_t val);
  void convertBoolNode(StringRef name, BoolAttr attr, bool negated = false);
  void convertI32Node(StringRef name, IntegerAttr attr);
  void convertFollowupNode(StringRef name, LoopAnnotationAttr attr);

  /// Conversion functions for each for each loop annotation sub-attribute.
  void convertLoopOptions(LoopVectorizeAttr options);
  void convertLoopOptions(LoopInterleaveAttr options);
  void convertLoopOptions(LoopUnrollAttr options);
  void convertLoopOptions(LoopUnrollAndJamAttr options);
  void convertLoopOptions(LoopLICMAttr options);
  void convertLoopOptions(LoopDistributeAttr options);
  void convertLoopOptions(LoopPipelineAttr options);
  void convertLoopOptions(LoopPeeledAttr options);
  void convertLoopOptions(LoopUnswitchAttr options);

  LoopAnnotationAttr attr;
  Operation *op;
  LoopAnnotationTranslation &loopAnnotationTranslation;
  llvm::LLVMContext &ctx;
  llvm::SmallVector<llvm::Metadata *> metadataNodes;
};
} // namespace

void LoopAnnotationConversion::addUnitNode(StringRef name) {
  metadataNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name)}));
}

void LoopAnnotationConversion::addUnitNode(StringRef name, BoolAttr attr) {
  if (attr && attr.getValue())
    addUnitNode(name);
}

void LoopAnnotationConversion::addI32NodeWithVal(StringRef name, uint32_t val) {
  llvm::Constant *cstValue = llvm::ConstantInt::get(
      llvm::IntegerType::get(ctx, /*NumBits=*/32), val, /*isSigned=*/false);
  metadataNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}

void LoopAnnotationConversion::convertBoolNode(StringRef name, BoolAttr attr,
                                               bool negated) {
  if (!attr)
    return;
  bool val = negated ^ attr.getValue();
  llvm::Constant *cstValue = llvm::ConstantInt::getBool(ctx, val);
  metadataNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name),
                              llvm::ConstantAsMetadata::get(cstValue)}));
}

void LoopAnnotationConversion::convertI32Node(StringRef name,
                                              IntegerAttr attr) {
  if (!attr)
    return;
  addI32NodeWithVal(name, attr.getInt());
}

void LoopAnnotationConversion::convertFollowupNode(StringRef name,
                                                   LoopAnnotationAttr attr) {
  if (!attr)
    return;

  llvm::MDNode *node =
      loopAnnotationTranslation.translateLoopAnnotation(attr, op);

  metadataNodes.push_back(
      llvm::MDNode::get(ctx, {llvm::MDString::get(ctx, name), node}));
}

void LoopAnnotationConversion::convertLoopOptions(LoopVectorizeAttr options) {
  convertBoolNode("llvm.loop.vectorize.enable", options.getDisable(), true);
  convertBoolNode("llvm.loop.vectorize.predicate.enable",
                  options.getPredicateEnable());
  convertBoolNode("llvm.loop.vectorize.scalable.enable",
                  options.getScalableEnable());
  convertI32Node("llvm.loop.vectorize.width", options.getWidth());
  convertFollowupNode("llvm.loop.vectorize.followup_vectorized",
                      options.getFollowupVectorized());
  convertFollowupNode("llvm.loop.vectorize.followup_epilogue",
                      options.getFollowupEpilogue());
  convertFollowupNode("llvm.loop.vectorize.followup_all",
                      options.getFollowupAll());
}

void LoopAnnotationConversion::convertLoopOptions(LoopInterleaveAttr options) {
  convertI32Node("llvm.loop.interleave.count", options.getCount());
}

void LoopAnnotationConversion::convertLoopOptions(LoopUnrollAttr options) {
  if (auto disable = options.getDisable())
    addUnitNode(disable.getValue() ? "llvm.loop.unroll.disable"
                                   : "llvm.loop.unroll.enable");
  convertI32Node("llvm.loop.unroll.count", options.getCount());
  convertBoolNode("llvm.loop.unroll.runtime.disable",
                  options.getRuntimeDisable());
  addUnitNode("llvm.loop.unroll.full", options.getFull());
  convertFollowupNode("llvm.loop.unroll.followup_unrolled",
                      options.getFollowupUnrolled());
  convertFollowupNode("llvm.loop.unroll.followup_remainder",
                      options.getFollowupRemainder());
  convertFollowupNode("llvm.loop.unroll.followup_all",
                      options.getFollowupAll());
}

void LoopAnnotationConversion::convertLoopOptions(
    LoopUnrollAndJamAttr options) {
  if (auto disable = options.getDisable())
    addUnitNode(disable.getValue() ? "llvm.loop.unroll_and_jam.disable"
                                   : "llvm.loop.unroll_and_jam.enable");
  convertI32Node("llvm.loop.unroll_and_jam.count", options.getCount());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_outer",
                      options.getFollowupOuter());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_inner",
                      options.getFollowupInner());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_outer",
                      options.getFollowupRemainderOuter());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_remainder_inner",
                      options.getFollowupRemainderInner());
  convertFollowupNode("llvm.loop.unroll_and_jam.followup_all",
                      options.getFollowupAll());
}

void LoopAnnotationConversion::convertLoopOptions(LoopLICMAttr options) {
  addUnitNode("llvm.licm.disable", options.getDisable());
  addUnitNode("llvm.loop.licm_versioning.disable",
              options.getVersioningDisable());
}

void LoopAnnotationConversion::convertLoopOptions(LoopDistributeAttr options) {
  convertBoolNode("llvm.loop.distribute.enable", options.getDisable(), true);
  convertFollowupNode("llvm.loop.distribute.followup_coincident",
                      options.getFollowupCoincident());
  convertFollowupNode("llvm.loop.distribute.followup_sequential",
                      options.getFollowupSequential());
  convertFollowupNode("llvm.loop.distribute.followup_fallback",
                      options.getFollowupFallback());
  convertFollowupNode("llvm.loop.distribute.followup_all",
                      options.getFollowupAll());
}

void LoopAnnotationConversion::convertLoopOptions(LoopPipelineAttr options) {
  convertBoolNode("llvm.loop.pipeline.disable", options.getDisable());
  convertI32Node("llvm.loop.pipeline.initiationinterval",
                 options.getInitiationinterval());
}

void LoopAnnotationConversion::convertLoopOptions(LoopPeeledAttr options) {
  convertI32Node("llvm.loop.peeled.count", options.getCount());
}

void LoopAnnotationConversion::convertLoopOptions(LoopUnswitchAttr options) {
  addUnitNode("llvm.loop.unswitch.partial.disable",
              options.getPartialDisable());
}

llvm::MDNode *LoopAnnotationConversion::convert() {

  // Reserve operand 0 for loop id self reference.
  auto dummy = llvm::MDNode::getTemporary(ctx, std::nullopt);
  metadataNodes.push_back(dummy.get());

  addUnitNode("llvm.loop.disable_nonforced", attr.getDisableNonforced());
  addUnitNode("llvm.loop.mustprogress", attr.getMustProgress());
  // "isvectorized" is encoded as an i32 value.
  if (BoolAttr isVectorized = attr.getIsVectorized())
    addI32NodeWithVal("llvm.loop.isvectorized", isVectorized.getValue());

  if (auto options = attr.getVectorize())
    convertLoopOptions(options);
  if (auto options = attr.getInterleave())
    convertLoopOptions(options);
  if (auto options = attr.getUnroll())
    convertLoopOptions(options);
  if (auto options = attr.getUnrollAndJam())
    convertLoopOptions(options);
  if (auto options = attr.getLicm())
    convertLoopOptions(options);
  if (auto options = attr.getDistribute())
    convertLoopOptions(options);
  if (auto options = attr.getPipeline())
    convertLoopOptions(options);
  if (auto options = attr.getPeeled())
    convertLoopOptions(options);
  if (auto options = attr.getUnswitch())
    convertLoopOptions(options);

  ArrayRef<SymbolRefAttr> parallelAccessGroups = attr.getParallelAccesses();
  if (!parallelAccessGroups.empty()) {
    SmallVector<llvm::Metadata *> parallelAccess;
    parallelAccess.push_back(
        llvm::MDString::get(ctx, "llvm.loop.parallel_accesses"));
    for (SymbolRefAttr accessGroupRef : parallelAccessGroups)
      parallelAccess.push_back(
          loopAnnotationTranslation.getAccessGroup(op, accessGroupRef));
    metadataNodes.push_back(llvm::MDNode::get(ctx, parallelAccess));
  }

  // Create loop options and set the first operand to itself.
  llvm::MDNode *loopMD = llvm::MDNode::get(ctx, metadataNodes);
  loopMD->replaceOperandWith(0, loopMD);

  return loopMD;
}

llvm::MDNode *
LoopAnnotationTranslation::translateLoopAnnotation(LoopAnnotationAttr attr,
                                                   Operation *op) {
  if (!attr)
    return nullptr;

  llvm::MDNode *loopMD = lookupLoopMetadata(attr);
  if (loopMD)
    return loopMD;

  loopMD =
      LoopAnnotationConversion(attr, op, *this, this->llvmModule.getContext())
          .convert();
  // Store a map from this Attribute to the LLVM metadata in case we
  // encounter it again.
  mapLoopMetadata(attr, loopMD);
  return loopMD;
}

LogicalResult LoopAnnotationTranslation::createAccessGroupMetadata() {
  mlirModule->walk([&](LLVM::MetadataOp metadatas) {
    metadatas.walk([&](LLVM::AccessGroupMetadataOp op) {
      llvm::MDNode *accessGroup =
          llvm::MDNode::getDistinct(llvmModule.getContext(), {});
      accessGroupMetadataMapping.insert({op, accessGroup});
    });
  });
  return success();
}

llvm::MDNode *
LoopAnnotationTranslation::getAccessGroup(Operation *op,
                                          SymbolRefAttr accessGroupRef) const {
  auto metadataName = accessGroupRef.getRootReference();
  auto accessGroupName = accessGroupRef.getLeafReference();
  auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
      op->getParentOp(), metadataName);
  auto *accessGroupOp =
      SymbolTable::lookupNearestSymbolFrom(metadataOp, accessGroupName);
  return accessGroupMetadataMapping.lookup(accessGroupOp);
}

llvm::MDNode *
LoopAnnotationTranslation::getAccessGroups(Operation *op,
                                           ArrayAttr accessGroupRefs) const {
  if (!accessGroupRefs || accessGroupRefs.empty())
    return nullptr;

  SmallVector<llvm::Metadata *> groupMDs;
  for (SymbolRefAttr groupRef : accessGroupRefs.getAsRange<SymbolRefAttr>())
    groupMDs.push_back(getAccessGroup(op, groupRef));
  if (groupMDs.size() == 1)
    return llvm::cast<llvm::MDNode>(groupMDs.front());
  return llvm::MDNode::get(llvmModule.getContext(), groupMDs);
}
