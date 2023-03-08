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
  // TODO: how can we get the lowering type for index type in the later pipeline
  // to be consistent? LLVM::StructureType does not allow index fields.
  auto sizeType = IntegerType::get(tp.getContext(), 64);
  auto lvlSizes = LLVM::LLVMArrayType::get(ctx, sizeType, lvlRank);
  auto memSizes = LLVM::LLVMArrayType::get(ctx, sizeType,
                                           getNumDataFieldsFromEncoding(enc));
  result.push_back(lvlSizes);
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

constexpr uint64_t kLvlSizePosInSpecifier = 0;
constexpr uint64_t kMemSizePosInSpecifier = 1;

class SpecifierStructBuilder : public StructBuilder {
private:
  Value extractField(OpBuilder &builder, Location loc,
                     ArrayRef<int64_t> indices) {
    return genCast(builder, loc,
                   builder.create<LLVM::ExtractValueOp>(loc, value, indices),
                   builder.getIndexType());
  }

  void insertField(OpBuilder &builder, Location loc, ArrayRef<int64_t> indices,
                   Value v) {
    value = builder.create<LLVM::InsertValueOp>(
        loc, value, genCast(builder, loc, v, builder.getIntegerType(64)),
        indices);
  }

public:
  explicit SpecifierStructBuilder(Value specifier) : StructBuilder(specifier) {
    assert(value);
  }

  // Undef value for level-sizes, all zero values for memory-sizes.
  static Value getInitValue(OpBuilder &builder, Location loc, Type structType);

  Value lvlSize(OpBuilder &builder, Location loc, Level lvl);
  void setLvlSize(OpBuilder &builder, Location loc, Level lvl, Value size);

  Value memSize(OpBuilder &builder, Location loc, FieldIndex fidx);
  void setMemSize(OpBuilder &builder, Location loc, FieldIndex fidx,
                  Value size);
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

/// Builds IR extracting the `lvl`-th level-size from the descriptor.
Value SpecifierStructBuilder::lvlSize(OpBuilder &builder, Location loc,
                                      Level lvl) {
  // This static_cast makes the narrowing of `lvl` explicit, as required
  // by the braces notation for the ctor.
  return extractField(
      builder, loc,
      ArrayRef<int64_t>{kLvlSizePosInSpecifier, static_cast<int64_t>(lvl)});
}

/// Builds IR inserting the `lvl`-th level-size into the descriptor.
void SpecifierStructBuilder::setLvlSize(OpBuilder &builder, Location loc,
                                        Level lvl, Value size) {
  // This static_cast makes the narrowing of `lvl` explicit, as required
  // by the braces notation for the ctor.
  insertField(
      builder, loc,
      ArrayRef<int64_t>{kLvlSizePosInSpecifier, static_cast<int64_t>(lvl)},
      size);
}

/// Builds IR extracting the `fidx`-th memory-size from the descriptor.
Value SpecifierStructBuilder::memSize(OpBuilder &builder, Location loc,
                                      FieldIndex fidx) {
  return extractField(builder, loc,
                      ArrayRef<int64_t>{kMemSizePosInSpecifier, fidx});
}

/// Builds IR inserting the `fidx`-th memory-size into the descriptor.
void SpecifierStructBuilder::setMemSize(OpBuilder &builder, Location loc,
                                        FieldIndex fidx, Value size) {
  insertField(builder, loc, ArrayRef<int64_t>{kMemSizePosInSpecifier, fidx},
              size);
}

} // namespace

//===----------------------------------------------------------------------===//
// The sparse storage specifier type converter (defined in Passes.h).
//===----------------------------------------------------------------------===//

StorageSpecifierToLLVMTypeConverter::StorageSpecifierToLLVMTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertSpecifier);
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
    if (op.getSpecifierKind() == StorageSpecifierKind::LvlSize) {
      assert(op.getLevel().has_value());
      v = Base::onLvlSize(rewriter, op, spec, op.getLevel().value());
    } else {
      auto enc = op.getSpecifier().getType().getEncoding();
      StorageLayout layout(enc);
      FieldIndex fidx =
          layout.getMemRefFieldIndex(op.getSpecifierKind(), op.getLevel());
      v = Base::onMemSize(rewriter, op, spec, fidx);
    }

    rewriter.replaceOp(op, v);
    return success();
  }
};

struct StorageSpecifierSetOpConverter
    : public SpecifierGetterSetterOpConverter<StorageSpecifierSetOpConverter,
                                              SetStorageSpecifierOp> {
  using SpecifierGetterSetterOpConverter::SpecifierGetterSetterOpConverter;
  static Value onLvlSize(OpBuilder &builder, SetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, Level lvl) {
    spec.setLvlSize(builder, op.getLoc(), lvl, op.getValue());
    return spec;
  }

  static Value onMemSize(OpBuilder &builder, SetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, FieldIndex fidx) {
    spec.setMemSize(builder, op.getLoc(), fidx, op.getValue());
    return spec;
  }
};

struct StorageSpecifierGetOpConverter
    : public SpecifierGetterSetterOpConverter<StorageSpecifierGetOpConverter,
                                              GetStorageSpecifierOp> {
  using SpecifierGetterSetterOpConverter::SpecifierGetterSetterOpConverter;
  static Value onLvlSize(OpBuilder &builder, GetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, Level lvl) {
    return spec.lvlSize(builder, op.getLoc(), lvl);
  }
  static Value onMemSize(OpBuilder &builder, GetStorageSpecifierOp op,
                         SpecifierStructBuilder &spec, FieldIndex fidx) {
    return spec.memSize(builder, op.getLoc(), fidx);
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
