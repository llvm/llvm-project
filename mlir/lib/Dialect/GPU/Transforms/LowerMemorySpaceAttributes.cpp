//===- LowerMemorySpaceAttributes.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of a pass that rewrites the IR so that uses of
/// `gpu::AddressSpaceAttr` in memref memory space annotations are replaced
/// with caller-specified numeric values.
///
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_GPULOWERMEMORYSPACEATTRIBUTESPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is considered as legal during memory space
/// attribute lowering.
static bool isLegalType(Type type) {
  if (auto memRefType = type.dyn_cast<BaseMemRefType>()) {
    return !memRefType.getMemorySpace()
                .isa_and_nonnull<gpu::AddressSpaceAttr>();
  }
  return true;
}

/// Returns true if the given `attr` is considered legal during memory space
/// attribute lowering.
static bool isLegalAttr(Attribute attr) {
  if (auto typeAttr = attr.dyn_cast<TypeAttr>())
    return isLegalType(typeAttr.getValue());
  return true;
}

/// Returns true if the given `op` is legal during memory space attribute
/// lowering.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::all_of(funcOp.getArgumentTypes(), isLegalType) &&
           llvm::all_of(funcOp.getResultTypes(), isLegalType) &&
           llvm::all_of(funcOp.getFunctionBody().getArgumentTypes(),
                        isLegalType);
  }

  auto attrs = llvm::map_range(op->getAttrs(), [](const NamedAttribute &attr) {
    return attr.getValue();
  });

  return llvm::all_of(op->getOperandTypes(), isLegalType) &&
         llvm::all_of(op->getResultTypes(), isLegalType) &&
         llvm::all_of(attrs, isLegalAttr);
}

void gpu::populateLowerMemorySpaceOpLegality(ConversionTarget &target) {
  target.markUnknownOpDynamicallyLegal(isLegalOp);
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

void mlir::gpu::populateMemorySpaceAttributeTypeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addConversion([mapping](Type type) -> Optional<Type> {
    auto subElementType = type.dyn_cast_or_null<SubElementTypeInterface>();
    if (!subElementType)
      return type;
    Type newType = subElementType.replaceSubElements(
        [mapping](Attribute attr) -> std::optional<Attribute> {
          auto memorySpaceAttr = attr.dyn_cast_or_null<gpu::AddressSpaceAttr>();
          if (!memorySpaceAttr)
            return std::nullopt;
          auto newValue = wrapNumericMemorySpace(
              attr.getContext(), mapping(memorySpaceAttr.getValue()));
          return newValue;
        });
    return newType;
  });
}

namespace {

/// Converts any op that has operands/results/attributes with numeric MemRef
/// memory spaces.
struct LowerMemRefAddressSpacePattern final : public ConversionPattern {
  LowerMemRefAddressSpacePattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> newAttrs;
    newAttrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs()) {
      if (auto typeAttr = attr.getValue().dyn_cast<TypeAttr>()) {
        auto newAttr = getTypeConverter()->convertType(typeAttr.getValue());
        newAttrs.emplace_back(attr.getName(), TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }
    }

    SmallVector<Type> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

    for (Region &region : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
} // namespace

void mlir::gpu::populateMemorySpaceLoweringPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<LowerMemRefAddressSpacePattern>(patterns.getContext(),
                                               typeConverter);
}

namespace {
class LowerMemorySpaceAttributesPass
    : public mlir::impl::GPULowerMemorySpaceAttributesPassBase<
          LowerMemorySpaceAttributesPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    ConversionTarget target(getContext());
    populateLowerMemorySpaceOpLegality(target);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    populateMemorySpaceAttributeTypeConversions(
        typeConverter, [this](AddressSpace space) -> unsigned {
          switch (space) {
          case AddressSpace::Global:
            return globalAddrSpace;
          case AddressSpace::Workgroup:
            return workgroupAddrSpace;
          case AddressSpace::Private:
            return privateAddrSpace;
          }
        });
    RewritePatternSet patterns(context);
    populateMemorySpaceLoweringPatterns(typeConverter, patterns);
    if (failed(applyFullConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
