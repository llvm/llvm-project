//===-- OneToNTypeConversion.cpp - Utils for 1:N type conversion-*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/OneToNTypeConversion.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

#include <unordered_map>

using namespace llvm;
using namespace mlir;

TypeRange OneToNTypeMapping::getConvertedTypes(unsigned originalTypeNo) const {
  TypeRange convertedTypes = getConvertedTypes();
  if (auto mapping = getInputMapping(originalTypeNo))
    return convertedTypes.slice(mapping->inputNo, mapping->size);
  return {};
}

ValueRange
OneToNTypeMapping::getConvertedValues(ValueRange convertedValues,
                                      unsigned originalValueNo) const {
  if (auto mapping = getInputMapping(originalValueNo))
    return convertedValues.slice(mapping->inputNo, mapping->size);
  return {};
}

void OneToNTypeMapping::convertLocation(
    Value originalValue, unsigned originalValueNo,
    llvm::SmallVectorImpl<Location> &result) const {
  if (auto mapping = getInputMapping(originalValueNo))
    result.append(mapping->size, originalValue.getLoc());
}

void OneToNTypeMapping::convertLocations(
    ValueRange originalValues, llvm::SmallVectorImpl<Location> &result) const {
  assert(originalValues.size() == getOriginalTypes().size());
  for (auto [i, value] : llvm::enumerate(originalValues))
    convertLocation(value, i, result);
}

static bool isIdentityConversion(Type originalType, TypeRange convertedTypes) {
  return convertedTypes.size() == 1 && convertedTypes[0] == originalType;
}

bool OneToNTypeMapping::hasNonIdentityConversion() const {
  // XXX: I think that the original types and the converted types are the same
  //      iff there was no non-identity type conversion. If that is true, the
  //      patterns could actually test whether there is anything useful to do
  //      without having access to the signature conversion.
  for (auto [i, originalType] : llvm::enumerate(originalTypes)) {
    TypeRange types = getConvertedTypes(i);
    if (!isIdentityConversion(originalType, types)) {
      assert(TypeRange(originalTypes) != getConvertedTypes());
      return true;
    }
  }
  assert(TypeRange(originalTypes) == getConvertedTypes());
  return false;
}

namespace {
enum class CastKind {
  // Casts block arguments in the target type back to the source type. (If
  // necessary, this cast becomes an argument materialization.)
  Argument,

  // Casts other values in the target type back to the source type. (If
  // necessary, this cast becomes a source materialization.)
  Source,

  // Casts values in the source type to the target type. (If necessary, this
  // cast becomes a target materialization.)
  Target
};
} // namespace

/// Mapping of enum values to string values.
StringRef getCastKindName(CastKind kind) {
  static const std::unordered_map<CastKind, StringRef> castKindNames = {
      {CastKind::Argument, "argument"},
      {CastKind::Source, "source"},
      {CastKind::Target, "target"}};
  return castKindNames.at(kind);
}

/// Attribute name that is used to annotate inserted unrealized casts with their
/// kind (source, argument, or target).
static const char *const castKindAttrName =
    "__one-to-n-type-conversion_cast-kind__";

/// Builds an `UnrealizedConversionCastOp` from the given inputs to the given
/// result types. Returns the result values of the cast.
static ValueRange buildUnrealizedCast(OpBuilder &builder, TypeRange resultTypes,
                                      ValueRange inputs, CastKind kind) {
  // Special case: 1-to-N conversion with N = 0. No need to build an
  // UnrealizedConversionCastOp because the op will always be dead.
  if (resultTypes.empty())
    return ValueRange();

  // Create cast.
  Location loc = builder.getUnknownLoc();
  if (!inputs.empty())
    loc = inputs.front().getLoc();
  auto castOp =
      builder.create<UnrealizedConversionCastOp>(loc, resultTypes, inputs);

  // Store cast kind as attribute.
  auto kindAttr = StringAttr::get(builder.getContext(), getCastKindName(kind));
  castOp->setAttr(castKindAttrName, kindAttr);

  return castOp->getResults();
}

/// Builds one `UnrealizedConversionCastOp` for each of the given original
/// values using the respective target types given in the provided conversion
/// mapping and returns the results of these casts. If the conversion mapping of
/// a value maps a type to itself (i.e., is an identity conversion), then no
/// cast is inserted and the original value is returned instead.
/// Note that these unrealized casts are different from target materializations
/// in that they are *always* inserted, even if they immediately fold away, such
/// that patterns always see valid intermediate IR, whereas materializations are
/// only used in the places where the unrealized casts *don't* fold away.
static SmallVector<Value>
buildUnrealizedForwardCasts(ValueRange originalValues,
                            OneToNTypeMapping &conversion,
                            RewriterBase &rewriter, CastKind kind) {

  // Convert each operand one by one.
  SmallVector<Value> convertedValues;
  convertedValues.reserve(conversion.getConvertedTypes().size());
  for (auto [idx, originalValue] : llvm::enumerate(originalValues)) {
    TypeRange convertedTypes = conversion.getConvertedTypes(idx);

    // Identity conversion: keep operand as is.
    if (isIdentityConversion(originalValue.getType(), convertedTypes)) {
      convertedValues.push_back(originalValue);
      continue;
    }

    // Non-identity conversion: materialize target types.
    ValueRange castResult =
        buildUnrealizedCast(rewriter, convertedTypes, originalValue, kind);
    convertedValues.append(castResult.begin(), castResult.end());
  }

  return convertedValues;
}

/// Builds one `UnrealizedConversionCastOp` for each sequence of the given
/// original values to one value of the type they originated from, i.e., a
/// "reverse" conversion from N converted values back to one value of the
/// original type, using the given (forward) type conversion. If a given value
/// was mapped to a value of the same type (i.e., the conversion in the mapping
/// is an identity conversion), then the "converted" value is returned without
/// cast.
/// Note that these unrealized casts are different from source materializations
/// in that they are *always* inserted, even if they immediately fold away, such
/// that patterns always see valid intermediate IR, whereas materializations are
/// only used in the places where the unrealized casts *don't* fold away.
static SmallVector<Value>
buildUnrealizedBackwardsCasts(ValueRange convertedValues,
                              const OneToNTypeMapping &typeConversion,
                              RewriterBase &rewriter) {
  assert(typeConversion.getConvertedTypes() == convertedValues.getTypes());

  // Create unrealized cast op for each converted result of the op.
  SmallVector<Value> recastValues;
  TypeRange originalTypes = typeConversion.getOriginalTypes();
  recastValues.reserve(originalTypes.size());
  auto convertedValueIt = convertedValues.begin();
  for (auto [idx, originalType] : llvm::enumerate(originalTypes)) {
    TypeRange convertedTypes = typeConversion.getConvertedTypes(idx);
    size_t numConvertedValues = convertedTypes.size();
    if (isIdentityConversion(originalType, convertedTypes)) {
      // Identity conversion: take result as is.
      recastValues.push_back(*convertedValueIt);
    } else {
      // Non-identity conversion: cast back to source type.
      ValueRange recastValue = buildUnrealizedCast(
          rewriter, originalType,
          ValueRange{convertedValueIt, convertedValueIt + numConvertedValues},
          CastKind::Source);
      assert(recastValue.size() == 1);
      recastValues.push_back(recastValue.front());
    }
    convertedValueIt += numConvertedValues;
  }

  return recastValues;
}

void OneToNPatternRewriter::replaceOp(Operation *op, ValueRange newValues,
                                      const OneToNTypeMapping &resultMapping) {
  // Create a cast back to the original types and replace the results of the
  // original op with those.
  assert(newValues.size() == resultMapping.getConvertedTypes().size());
  assert(op->getResultTypes() == resultMapping.getOriginalTypes());
  PatternRewriter::InsertionGuard g(*this);
  setInsertionPointAfter(op);
  SmallVector<Value> castResults =
      buildUnrealizedBackwardsCasts(newValues, resultMapping, *this);
  replaceOp(op, castResults);
}

Block *OneToNPatternRewriter::applySignatureConversion(
    Block *block, OneToNTypeMapping &argumentConversion) {
  PatternRewriter::InsertionGuard g(*this);

  // Split the block at the beginning to get a new block to use for the
  // updated signature.
  SmallVector<Location> locs;
  argumentConversion.convertLocations(block->getArguments(), locs);
  Block *newBlock =
      createBlock(block, argumentConversion.getConvertedTypes(), locs);
  replaceAllUsesWith(block, newBlock);

  // Create necessary casts in new block.
  SmallVector<Value> castResults;
  for (auto [i, arg] : llvm::enumerate(block->getArguments())) {
    TypeRange convertedTypes = argumentConversion.getConvertedTypes(i);
    ValueRange newArgs =
        argumentConversion.getConvertedValues(newBlock->getArguments(), i);
    if (isIdentityConversion(arg.getType(), convertedTypes)) {
      // Identity conversion: take argument as is.
      assert(newArgs.size() == 1);
      castResults.push_back(newArgs.front());
    } else {
      // Non-identity conversion: cast the converted arguments to the original
      // type.
      PatternRewriter::InsertionGuard g(*this);
      setInsertionPointToStart(newBlock);
      ValueRange castResult = buildUnrealizedCast(*this, arg.getType(), newArgs,
                                                  CastKind::Argument);
      assert(castResult.size() == 1);
      castResults.push_back(castResult.front());
    }
  }

  // Merge old block into new block such that we only have the latter with the
  // new signature.
  mergeBlocks(block, newBlock, castResults);

  return newBlock;
}

LogicalResult
OneToNConversionPattern::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const {
  auto *typeConverter = getTypeConverter();

  // Construct conversion mapping for results.
  Operation::result_type_range originalResultTypes = op->getResultTypes();
  OneToNTypeMapping resultMapping(originalResultTypes);
  if (failed(typeConverter->convertSignatureArgs(originalResultTypes,
                                                 resultMapping)))
    return failure();

  // Construct conversion mapping for operands.
  Operation::operand_type_range originalOperandTypes = op->getOperandTypes();
  OneToNTypeMapping operandMapping(originalOperandTypes);
  if (failed(typeConverter->convertSignatureArgs(originalOperandTypes,
                                                 operandMapping)))
    return failure();

  // Cast operands to target types.
  SmallVector<Value> convertedOperands = buildUnrealizedForwardCasts(
      op->getOperands(), operandMapping, rewriter, CastKind::Target);

  // Create a `OneToNPatternRewriter` for the pattern, which provides additional
  // functionality.
  // TODO(ingomueller): I guess it would be better to use only one rewriter
  //                    throughout the whole pass, but that would require to
  //                    drive the pattern application ourselves, which is a lot
  //                    of additional boilerplate code. This seems to work fine,
  //                    so I leave it like this for the time being.
  OneToNPatternRewriter oneToNPatternRewriter(rewriter.getContext(),
                                              rewriter.getListener());
  oneToNPatternRewriter.restoreInsertionPoint(rewriter.saveInsertionPoint());

  // Apply actual pattern.
  if (failed(matchAndRewrite(op, oneToNPatternRewriter, operandMapping,
                             resultMapping, convertedOperands)))
    return failure();

  return success();
}

namespace mlir {

// This function applies the provided patterns using
// `applyPatternsGreedily` and then replaces all newly inserted
// `UnrealizedConversionCastOps` that haven't folded away. ("Backward" casts
// from target to source types inserted by a `OneToNConversionPattern` normally
// fold away with the "forward" casts from source to target types inserted by
// the next pattern.) To understand which casts are "newly inserted", all casts
// inserted by this pass are annotated with a string attribute that also
// documents which kind of the cast (source, argument, or target).
LogicalResult
applyPartialOneToNConversion(Operation *op, TypeConverter &typeConverter,
                             const FrozenRewritePatternSet &patterns) {
#ifndef NDEBUG
  // Remember existing unrealized casts. This data structure is only used in
  // asserts; building it only for that purpose may be an overkill.
  SmallSet<UnrealizedConversionCastOp, 4> existingCasts;
  op->walk([&](UnrealizedConversionCastOp castOp) {
    assert(!castOp->hasAttr(castKindAttrName));
    existingCasts.insert(castOp);
  });
#endif // NDEBUG

  // Apply provided conversion patterns.
  if (failed(applyPatternsGreedily(op, patterns))) {
    emitError(op->getLoc()) << "failed to apply conversion patterns";
    return failure();
  }

  // Find all unrealized casts inserted by the pass that haven't folded away.
  SmallVector<UnrealizedConversionCastOp> worklist;
  op->walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->hasAttr(castKindAttrName)) {
      assert(!existingCasts.contains(castOp));
      worklist.push_back(castOp);
    }
  });

  // Replace new casts with user materializations.
  IRRewriter rewriter(op->getContext());
  for (UnrealizedConversionCastOp castOp : worklist) {
    TypeRange resultTypes = castOp->getResultTypes();
    ValueRange operands = castOp->getOperands();
    StringRef castKind =
        castOp->getAttrOfType<StringAttr>(castKindAttrName).getValue();
    rewriter.setInsertionPoint(castOp);

#ifndef NDEBUG
    // Determine whether operands or results are already legal to test some
    // assumptions for the different kind of materializations. These properties
    // are only used it asserts and it may be overkill to compute them.
    bool areOperandTypesLegal = llvm::all_of(
        operands.getTypes(), [&](Type t) { return typeConverter.isLegal(t); });
    bool areResultsTypesLegal = llvm::all_of(
        resultTypes, [&](Type t) { return typeConverter.isLegal(t); });
#endif // NDEBUG

    // Add materialization and remember materialized results.
    SmallVector<Value> materializedResults;
    if (castKind == getCastKindName(CastKind::Target)) {
      // Target materialization.
      assert(!areOperandTypesLegal && areResultsTypesLegal &&
             operands.size() == 1 && "found unexpected target cast");
      materializedResults = typeConverter.materializeTargetConversion(
          rewriter, castOp->getLoc(), resultTypes, operands.front());
      if (materializedResults.empty()) {
        emitError(castOp->getLoc())
            << "failed to create target materialization";
        return failure();
      }
    } else {
      // Source and argument materializations.
      assert(areOperandTypesLegal && !areResultsTypesLegal &&
             resultTypes.size() == 1 && "found unexpected cast");
      std::optional<Value> maybeResult;
      if (castKind == getCastKindName(CastKind::Source)) {
        // Source materialization.
        maybeResult = typeConverter.materializeSourceConversion(
            rewriter, castOp->getLoc(), resultTypes.front(),
            castOp.getOperands());
      } else {
        // Argument materialization.
        assert(castKind == getCastKindName(CastKind::Argument) &&
               "unexpected value of cast kind attribute");
        assert(llvm::all_of(operands, llvm::IsaPred<BlockArgument>));
        maybeResult = typeConverter.materializeArgumentConversion(
            rewriter, castOp->getLoc(), resultTypes.front(),
            castOp.getOperands());
      }
      if (!maybeResult.has_value() || !maybeResult.value()) {
        emitError(castOp->getLoc())
            << "failed to create " << castKind << " materialization";
        return failure();
      }
      materializedResults = {maybeResult.value()};
    }

    // Replace the cast with the result of the materialization.
    rewriter.replaceOp(castOp, materializedResults);
  }

  return success();
}

namespace {
class FunctionOpInterfaceSignatureConversion : public OneToNConversionPattern {
public:
  FunctionOpInterfaceSignatureConversion(StringRef functionLikeOpName,
                                         MLIRContext *ctx,
                                         const TypeConverter &converter)
      : OneToNConversionPattern(converter, functionLikeOpName, /*benefit=*/1,
                                ctx) {}

  LogicalResult matchAndRewrite(Operation *op, OneToNPatternRewriter &rewriter,
                                const OneToNTypeMapping &operandMapping,
                                const OneToNTypeMapping &resultMapping,
                                ValueRange convertedOperands) const override {
    auto funcOp = cast<FunctionOpInterface>(op);
    auto *typeConverter = getTypeConverter();

    // Construct mapping for function arguments.
    OneToNTypeMapping argumentMapping(funcOp.getArgumentTypes());
    if (failed(typeConverter->convertSignatureArgs(funcOp.getArgumentTypes(),
                                                   argumentMapping)))
      return failure();

    // Construct mapping for function results.
    OneToNTypeMapping funcResultMapping(funcOp.getResultTypes());
    if (failed(typeConverter->convertSignatureArgs(funcOp.getResultTypes(),
                                                   funcResultMapping)))
      return failure();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!argumentMapping.hasNonIdentityConversion() &&
        !funcResultMapping.hasNonIdentityConversion())
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     argumentMapping.getConvertedTypes(),
                                     funcResultMapping.getConvertedTypes());
    rewriter.modifyOpInPlace(op, [&] { funcOp.setType(newType); });

    // Update block signatures.
    if (!funcOp.isExternal()) {
      Region *region = &funcOp.getFunctionBody();
      Block *block = &region->front();
      rewriter.applySignatureConversion(block, argumentMapping);
    }

    return success();
  }
};
} // namespace

void populateOneToNFunctionOpInterfaceTypeConversionPattern(
    StringRef functionLikeOpName, const TypeConverter &converter,
    RewritePatternSet &patterns) {
  patterns.add<FunctionOpInterfaceSignatureConversion>(
      functionLikeOpName, patterns.getContext(), converter);
}
} // namespace mlir
