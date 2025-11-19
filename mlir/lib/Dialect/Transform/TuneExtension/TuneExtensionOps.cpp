//===- TuneExtensionOps.cpp - Tune extension for the Transform dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.h"

using namespace mlir;

static ParseResult parseAlternativesOpSelectedRegion(
    OpAsmParser &parser, IntegerAttr &selectedRegionAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &selectedRegionParam);

static void printAlternativesOpSelectedRegion(OpAsmPrinter &printer,
                                              Operation *op,
                                              IntegerAttr selectedRegionAttr,
                                              Value selectedRegionParam);

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.cpp.inc"

#define DEBUG_TYPE "transform-tune"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

//===----------------------------------------------------------------------===//
// KnobOp
//===----------------------------------------------------------------------===//

void transform::tune::KnobOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform::tune::KnobOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  if (getSelected()) {
    results.setParams(llvm::cast<OpResult>(getResult()), *getSelected());
    return DiagnosedSilenceableFailure::success();
  }

  return emitDefiniteFailure()
         << "non-deterministic choice " << getName()
         << " is only resolved through providing a `selected` attr";
}

LogicalResult transform::tune::KnobOp::verify() {
  if (auto selected = getSelected()) {
    if (auto optionsArray = dyn_cast<ArrayAttr>(getOptions())) {
      if (!llvm::is_contained(optionsArray, selected))
        return emitOpError("provided `selected` attribute is not an element of "
                           "`options` array of attributes");
    } else
      LLVM_DEBUG(DBGS() << "cannot verify `selected` attribute " << selected
                        << " is an element of `options` attribute "
                        << getOptions());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AlternativesOp
//===----------------------------------------------------------------------===//

static ParseResult parseAlternativesOpSelectedRegion(
    OpAsmParser &parser, IntegerAttr &selectedRegionAttr,
    std::optional<OpAsmParser::UnresolvedOperand> &selectedRegionParam) {
  size_t selectedRegionIdx;
  OptionalParseResult attrParseRes =
      parser.parseOptionalInteger(selectedRegionIdx);
  if (attrParseRes.has_value()) {
    if (failed(*attrParseRes))
      return failure();

    selectedRegionAttr = parser.getBuilder().getIndexAttr(selectedRegionIdx);
    return success();
  }

  OpAsmParser::UnresolvedOperand param;
  auto paramParseRes = parser.parseOptionalOperand(param);
  if (paramParseRes.has_value()) {
    if (failed(*paramParseRes))
      return failure();

    selectedRegionParam = param;
    return success();
  }

  return parser.emitError(parser.getCurrentLocation())
         << "expected either an integer attribute or a transform.param operand";
}

static void printAlternativesOpSelectedRegion(OpAsmPrinter &printer,
                                              Operation *op,
                                              IntegerAttr selectedRegionAttr,
                                              Value selectedRegionParam) {
  if (selectedRegionAttr)
    printer << selectedRegionAttr.getValue();
  if (selectedRegionParam)
    printer << selectedRegionParam;
}

OperandRange transform::tune::AlternativesOp::getEntrySuccessorOperands(
    RegionSuccessor successor) {
  // No operands will be forwarded to the region(s).
  return getOperands().slice(0, 0);
}

void transform::tune::AlternativesOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent())
    if (auto selectedRegionIdx = getSelectedRegionAttr())
      regions.emplace_back(
          &getAlternatives()[selectedRegionIdx->getSExtValue()],
          Block::BlockArgListType());
    else
      for (Region &alternative : getAlternatives())
        regions.emplace_back(&alternative, Block::BlockArgListType());
  else
    regions.emplace_back(getOperation(), getOperation()->getResults());
}

void transform::tune::AlternativesOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  (void)operands;
  bounds.reserve(getNumRegions());

  if (auto selectedRegionIdx = getSelectedRegionAttr()) {
    bounds.resize(getNumRegions(), InvocationBounds(0, 0));
    bounds[selectedRegionIdx->getSExtValue()] = InvocationBounds(1, 1);
  } else {
    bounds.resize(getNumRegions(), InvocationBounds(0, 1));
  }
}

void transform::tune::AlternativesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getSelectedRegionParamMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  // TODO: should effects from regions be forwarded?
}

DiagnosedSilenceableFailure
transform::tune::AlternativesOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  std::optional<size_t> selectedRegionIdx;

  if (auto selectedRegionAttr = getSelectedRegionAttr())
    selectedRegionIdx = selectedRegionAttr->getSExtValue();

  if (Value selectedRegionParam = getSelectedRegionParam()) {
    ArrayRef<Attribute> associatedAttrs = state.getParams(selectedRegionParam);
    IntegerAttr selectedRegionAttr;
    if (associatedAttrs.size() != 1 ||
        !(selectedRegionAttr = dyn_cast<IntegerAttr>(associatedAttrs[0])))
      return emitDefiniteFailure()
             << "param should hold exactly one integer attribute, got: "
             << associatedAttrs[0];
    selectedRegionIdx = selectedRegionAttr.getValue().getSExtValue();
  }

  if (!selectedRegionIdx)
    return emitDefiniteFailure() << "non-deterministic choice " << getName()
                                 << " is only resolved through providing a "
                                    "`selected_region` attr/param";

  if (*selectedRegionIdx < 0 || *selectedRegionIdx >= getNumRegions())
    return emitDefiniteFailure()
           << "'selected_region' attribute/param specifies region at index "
           << *selectedRegionIdx << " while op has only " << getNumRegions()
           << " regions";

  Region &selectedRegion = getRegion(*selectedRegionIdx);
  auto scope = state.make_region_scope(selectedRegion);
  Block &block = selectedRegion.front();
  // Apply the region's ops one by one.
  for (Operation &transform : block.without_terminator()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<transform::TransformOpInterface>(transform));
    if (result.isDefiniteFailure())
      return result;

    if (result.isSilenceableFailure()) {
      for (const auto &res : getResults())
        results.set(res, {});
      return result;
    }
  }
  // Forward the operation mapping for values yielded from the region to the
  // values produced by the alternatives op.
  transform::detail::forwardTerminatorOperands(&block, state, results);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::tune::AlternativesOp::verify() {
  for (auto *region : getRegions()) {
    auto yieldTerminator =
        llvm::dyn_cast_if_present<transform::YieldOp>(region->front().back());
    if (!yieldTerminator)
      return emitOpError() << "expected '"
                           << transform::YieldOp::getOperationName()
                           << "' as terminator";

    if (yieldTerminator->getNumOperands() != getNumResults())
      return yieldTerminator.emitOpError()
             << "expected terminator to have as many operands as the parent op "
                "has results";

    for (auto [i, operandType, resultType] : llvm::zip_equal(
             llvm::seq<unsigned>(0, yieldTerminator->getNumOperands()),
             yieldTerminator->getOperands().getType(), getResultTypes())) {
      if (operandType == resultType)
        continue;
      return yieldTerminator.emitOpError()
             << "the type of the terminator operand #" << i
             << " must match the type of the corresponding parent op result ("
             << operandType << " vs " << resultType << ")";
    }
  }

  if (auto selectedRegionAttr = getSelectedRegionAttr()) {
    size_t regionIdx = selectedRegionAttr->getSExtValue();
    if (regionIdx < 0 || regionIdx >= getNumRegions())
      return emitOpError()
             << "'selected_region' attribute specifies region at index "
             << regionIdx << " while op has only " << getNumRegions()
             << " regions";
  }

  return success();
}
