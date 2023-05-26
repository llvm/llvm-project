//===- SparseStorageSpecifierToLLVM.cpp - convert specifier to llvm -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorStorageLayout.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

#include <optional>

using namespace mlir;
using namespace sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

static SmallVector<Type, 4> getSpecifierFields(StorageSpecifierType tp) {
  MLIRContext *ctx = tp.getContext();
  auto enc = tp.getEncoding();
  const Level lvlRank = enc.getLvlRank();

  SmallVector<Type, 4> result;
  // TODO: how can we get the lowering type for index type in the later pipeline
  // to be consistent? LLVM::StructureType does not allow index fields.
  auto sizeType = IntegerType::get(tp.getContext(), 64);
  auto lvlSizes = LLVM::LLVMArrayType::get(ctx, sizeType, lvlRank);
  auto memSizes = LLVM::LLVMArrayType::get(ctx, sizeType,
                                           getNumDataFieldsFromEncoding(enc));
  result.push_back(lvlSizes);
  result.push_back(memSizes);

  if (enc.isSlice()) {
    // Extra fields are required for the slice information.
    auto dimOffset = LLVM::LLVMArrayType::get(ctx, sizeType, lvlRank);
    auto dimStride = LLVM::LLVMArrayType::get(ctx, sizeType, lvlRank);

    result.push_back(dimOffset);
    result.push_back(dimStride);
  }

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
constexpr uint64_t kDimOffsetPosInSpecifier = 2;
constexpr uint64_t kDimStridePosInSpecifier = 3;

class SpecifierStructBuilder : public StructBuilder {
private:
  Value extractField(OpBuilder &builder, Location loc,
                     ArrayRef<int64_t> indices) const {
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

  // Undef value for dimension sizes, all zero value for memory sizes.
  static Value getInitValue(OpBuilder &builder, Location loc, Type structType,
                            Value source);

  Value lvlSize(OpBuilder &builder, Location loc, Level lvl) const;
  void setLvlSize(OpBuilder &builder, Location loc, Level lvl, Value size);

  Value dimOffset(OpBuilder &builder, Location loc, Dimension dim) const;
  void setDimOffset(OpBuilder &builder, Location loc, Dimension dim,
                    Value size);

  Value dimStride(OpBuilder &builder, Location loc, Dimension dim) const;
  void setDimStride(OpBuilder &builder, Location loc, Dimension dim,
                    Value size);

  Value memSize(OpBuilder &builder, Location loc, FieldIndex fidx) const;
  void setMemSize(OpBuilder &builder, Location loc, FieldIndex fidx,
                  Value size);

  Value memSizeArray(OpBuilder &builder, Location loc) const;
  void setMemSizeArray(OpBuilder &builder, Location loc, Value array);
};

Value SpecifierStructBuilder::getInitValue(OpBuilder &builder, Location loc,
                                           Type structType, Value source) {
  Value metaData = builder.create<LLVM::UndefOp>(loc, structType);
  SpecifierStructBuilder md(metaData);
  if (!source) {
    auto memSizeArrayType =
        cast<LLVM::LLVMArrayType>(cast<LLVM::LLVMStructType>(structType)
                                      .getBody()[kMemSizePosInSpecifier]);

    Value zero = constantZero(builder, loc, memSizeArrayType.getElementType());
    // Fill memSizes array with zero.
    for (int i = 0, e = memSizeArrayType.getNumElements(); i < e; i++)
      md.setMemSize(builder, loc, i, zero);
  } else {
    // We copy non-slice information (memory sizes array) from source
    SpecifierStructBuilder sourceMd(source);
    md.setMemSizeArray(builder, loc, sourceMd.memSizeArray(builder, loc));
  }
  return md;
}

/// Builds IR extracting the pos-th offset from the descriptor.
Value SpecifierStructBuilder::dimOffset(OpBuilder &builder, Location loc,
                                        Dimension dim) const {
  return extractField(
      builder, loc,
      ArrayRef<int64_t>{kDimOffsetPosInSpecifier, static_cast<int64_t>(dim)});
}

/// Builds IR inserting the pos-th offset into the descriptor.
void SpecifierStructBuilder::setDimOffset(OpBuilder &builder, Location loc,
                                          Dimension dim, Value size) {
  insertField(
      builder, loc,
      ArrayRef<int64_t>{kDimOffsetPosInSpecifier, static_cast<int64_t>(dim)},
      size);
}

/// Builds IR extracting the `lvl`-th level-size from the descriptor.
Value SpecifierStructBuilder::lvlSize(OpBuilder &builder, Location loc,
                                      Level lvl) const {
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

/// Builds IR extracting the pos-th stride from the descriptor.
Value SpecifierStructBuilder::dimStride(OpBuilder &builder, Location loc,
                                        Dimension dim) const {
  return extractField(
      builder, loc,
      ArrayRef<int64_t>{kDimStridePosInSpecifier, static_cast<int64_t>(dim)});
}

/// Builds IR inserting the pos-th stride into the descriptor.
void SpecifierStructBuilder::setDimStride(OpBuilder &builder, Location loc,
                                          Dimension dim, Value size) {
  insertField(
      builder, loc,
      ArrayRef<int64_t>{kDimStridePosInSpecifier, static_cast<int64_t>(dim)},
      size);
}

/// Builds IR extracting the pos-th memory size into the descriptor.
Value SpecifierStructBuilder::memSize(OpBuilder &builder, Location loc,
                                      FieldIndex fidx) const {
  return extractField(
      builder, loc,
      ArrayRef<int64_t>{kMemSizePosInSpecifier, static_cast<int64_t>(fidx)});
}

/// Builds IR inserting the `fidx`-th memory-size into the descriptor.
void SpecifierStructBuilder::setMemSize(OpBuilder &builder, Location loc,
                                        FieldIndex fidx, Value size) {
  insertField(
      builder, loc,
      ArrayRef<int64_t>{kMemSizePosInSpecifier, static_cast<int64_t>(fidx)},
      size);
}

/// Builds IR extracting the memory size array from the descriptor.
Value SpecifierStructBuilder::memSizeArray(OpBuilder &builder,
                                           Location loc) const {
  return builder.create<LLVM::ExtractValueOp>(loc, value,
                                              kMemSizePosInSpecifier);
}

/// Builds IR inserting the memory size array into the descriptor.
void SpecifierStructBuilder::setMemSizeArray(OpBuilder &builder, Location loc,
                                             Value array) {
  value = builder.create<LLVM::InsertValueOp>(loc, value, array,
                                              kMemSizePosInSpecifier);
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
    switch (op.getSpecifierKind()) {
    case StorageSpecifierKind::LvlSize: {
      Value v = Base::onLvlSize(rewriter, op, spec, (*op.getLevel()));
      rewriter.replaceOp(op, v);
      return success();
    }
    case StorageSpecifierKind::DimOffset: {
      Value v = Base::onDimOffset(rewriter, op, spec, (*op.getLevel()));
      rewriter.replaceOp(op, v);
      return success();
    }
    case StorageSpecifierKind::DimStride: {
      Value v = Base::onDimStride(rewriter, op, spec, (*op.getLevel()));
      rewriter.replaceOp(op, v);
      return success();
    }
    case StorageSpecifierKind::CrdMemSize:
    case StorageSpecifierKind::PosMemSize:
    case StorageSpecifierKind::ValMemSize: {
      auto enc = op.getSpecifier().getType().getEncoding();
      StorageLayout layout(enc);
      std::optional<unsigned> lvl;
      if (op.getLevel())
        lvl = (*op.getLevel());
      unsigned idx =
          layout.getMemRefFieldIndex(toFieldKind(op.getSpecifierKind()), lvl);
      Value v = Base::onMemSize(rewriter, op, spec, idx);
      rewriter.replaceOp(op, v);
      return success();
    }
    }
    llvm_unreachable("unrecognized specifer kind");
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

  static Value onDimOffset(OpBuilder &builder, SetStorageSpecifierOp op,
                           SpecifierStructBuilder &spec, Dimension d) {
    spec.setDimOffset(builder, op.getLoc(), d, op.getValue());
    return spec;
  }

  static Value onDimStride(OpBuilder &builder, SetStorageSpecifierOp op,
                           SpecifierStructBuilder &spec, Dimension d) {
    spec.setDimStride(builder, op.getLoc(), d, op.getValue());
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

  static Value onDimOffset(OpBuilder &builder, GetStorageSpecifierOp op,
                           const SpecifierStructBuilder &spec, Dimension d) {
    return spec.dimOffset(builder, op.getLoc(), d);
  }

  static Value onDimStride(OpBuilder &builder, GetStorageSpecifierOp op,
                           const SpecifierStructBuilder &spec, Dimension d) {
    return spec.dimStride(builder, op.getLoc(), d);
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
    rewriter.replaceOp(
        op, SpecifierStructBuilder::getInitValue(
                rewriter, op.getLoc(), llvmType, adaptor.getSource()));
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
