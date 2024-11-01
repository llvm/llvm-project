//===- SparseStorageSpecifierToLLVM.cpp - convert specifier to llvm -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "SparseTensorStorageLayout.h"

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include <optional>

using namespace mlir;
using namespace sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

static SmallVector<Type, 2> getSpecifierFields(StorageSpecifierType tp) {
  MLIRContext *ctx = tp.getContext();
  auto enc = tp.getEncoding();
  const Level lvlRank = enc.getLvlRank();

  SmallVector<Type, 2> result;
  auto indexType = tp.getSizesType();
  auto dimSizes = LLVM::LLVMArrayType::get(ctx, indexType, lvlRank);
  auto memSizes = LLVM::LLVMArrayType::get(ctx, indexType,
                                           getNumDataFieldsFromEncoding(enc));
  result.push_back(dimSizes);
  result.push_back(memSizes);
  return result;
}

static Type convertSpecifier(StorageSpecifierType tp) {
  return LLVM::LLVMStructType::getLiteral(tp.getContext(),
                                          getSpecifierFields(tp));
}

//===----------------------------------------------------------------------===//
// Specifier struct builder.
//===----------------------------------------------------------------------===//

constexpr uint64_t kDimSizePosInSpecifier = 0;
constexpr uint64_t kMemSizePosInSpecifier = 1;

class SpecifierStructBuilder : public StructBuilder {
public:
  explicit SpecifierStructBuilder(Value specifier) : StructBuilder(specifier) {
    assert(value);
  }

  // Undef value for dimension sizes, all zero value for memory sizes.
  static Value getInitValue(OpBuilder &builder, Location loc, Type structType);

  Value dimSize(OpBuilder &builder, Location loc, unsigned dim);
  void setDimSize(OpBuilder &builder, Location loc, unsigned dim, Value size);

  Value memSize(OpBuilder &builder, Location loc, unsigned pos);
  void setMemSize(OpBuilder &builder, Location loc, unsigned pos, Value size);
};

Value SpecifierStructBuilder::getInitValue(OpBuilder &builder, Location loc,
                                           Type structType) {
  Value metaData = builder.create<LLVM::UndefOp>(loc, structType);
  SpecifierStructBuilder md(metaData);
  auto memSizeArrayType = structType.cast<LLVM::LLVMStructType>()
                              .getBody()[kMemSizePosInSpecifier]
                              .cast<LLVM::LLVMArrayType>();

  Value zero = constantZero(builder, loc, memSizeArrayType.getElementType());
  // Fill memSizes array with zero.
  for (int i = 0, e = memSizeArrayType.getNumElements(); i < e; i++)
    md.setMemSize(builder, loc, i, zero);

  return md;
}

/// Builds IR inserting the pos-th size into the descriptor.
Value SpecifierStructBuilder::dimSize(OpBuilder &builder, Location loc,
                                      unsigned dim) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, value, ArrayRef<int64_t>({kDimSizePosInSpecifier, dim}));
}

/// Builds IR inserting the pos-th size into the descriptor.
void SpecifierStructBuilder::setDimSize(OpBuilder &builder, Location loc,
                                        unsigned dim, Value size) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, value, size, ArrayRef<int64_t>({kDimSizePosInSpecifier, dim}));
}

/// Builds IR extracting the pos-th memory size into the descriptor.
Value SpecifierStructBuilder::memSize(OpBuilder &builder, Location loc,
                                      unsigned pos) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, value, ArrayRef<int64_t>({kMemSizePosInSpecifier, pos}));
}

/// Builds IR inserting the pos-th memory size into the descriptor.
void SpecifierStructBuilder::setMemSize(OpBuilder &builder, Location loc,
                                        unsigned pos, Value size) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, value, size, ArrayRef<int64_t>({kMemSizePosInSpecifier, pos}));
}

} // namespace

//===----------------------------------------------------------------------===//
// The sparse storage specifier type converter (defined in Passes.h).
//===----------------------------------------------------------------------===//

StorageSpecifierToLLVMTypeConverter::StorageSpecifierToLLVMTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([](StorageSpecifierType tp) { return convertSpecifier(tp); });
}

//===----------------------------------------------------------------------===//
// Storage specifier conversion rules.
//===----------------------------------------------------------------------===//

template <typename Base, typename SourceOp>
class SpecifierGetterSetterOpConverter : public OpConversionPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SpecifierStructBuilder spec(adaptor.getSpecifier());
    Value v;
    if (op.getSpecifierKind() == StorageSpecifierKind::DimSize) {
      v = Base::onDimSize(rewriter, op, spec,
                          op.getDim().value().getZExtValue());
    } else {
      auto enc = op.getSpecifier().getType().getEncoding();
      StorageLayout layout(enc);
      std::optional<unsigned> dim;
      if (op.getDim())
        dim = op.getDim().value().getZExtValue();
      unsigned idx = layout.getMemRefFieldIndex(op.getSpecifierKind(), dim);
      v = Base::onMemSize(rewriter, op, spec, idx);
    }

    rewriter.replaceOp(op, v);
    return success();
  }
};

struct StorageSpecifierSetOpConverter
    : public SpecifierGetterSetterOpConverter<StorageSpecifierSetOpConverter,
                                              SetStorageSpecifierOp> {
  using SpecifierGetterSetterOpConverter::SpecifierGetterSetterOpConverter;
  static Value onDimSize(OpBuilder &builder, SetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, unsigned d) {
    spec.setDimSize(builder, op.getLoc(), d, op.getValue());
    return spec;
  }

  static Value onMemSize(OpBuilder &builder, SetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, unsigned i) {
    spec.setMemSize(builder, op.getLoc(), i, op.getValue());
    return spec;
  }
};

struct StorageSpecifierGetOpConverter
    : public SpecifierGetterSetterOpConverter<StorageSpecifierGetOpConverter,
                                              GetStorageSpecifierOp> {
  using SpecifierGetterSetterOpConverter::SpecifierGetterSetterOpConverter;
  static Value onDimSize(OpBuilder &builder, GetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, unsigned d) {
    return spec.dimSize(builder, op.getLoc(), d);
  }
  static Value onMemSize(OpBuilder &builder, GetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, unsigned i) {
    return spec.memSize(builder, op.getLoc(), i);
  }
};

struct StorageSpecifierInitOpConverter
    : public OpConversionPattern<StorageSpecifierInitOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StorageSpecifierInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOp(op, SpecifierStructBuilder::getInitValue(
                               rewriter, op.getLoc(), llvmType));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

void mlir::populateStorageSpecifierToLLVMPatterns(TypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  patterns.add<StorageSpecifierGetOpConverter, StorageSpecifierSetOpConverter,
               StorageSpecifierInitOpConverter>(converter,
                                                patterns.getContext());
}
