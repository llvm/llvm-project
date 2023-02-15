//===- SparseTensorType.h - Wrapper around RankedTensorType -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the `SparseTensorType` wrapper class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
/// A wrapper around `RankedTensorType`, which has three goals:
///
/// (1) To provide a uniform API for querying aspects of sparse-tensor
/// types; in particular, to make the "dimension" vs "level" distinction
/// overt (i.e., explicit everywhere).  Thus, throughout the sparse-compiler
/// this class should be preferred over using `RankedTensorType` or
/// `ShapedType` directly, since the methods of the latter do not make
/// the "dimension" vs "level" distinction overt.
///
/// (2) To provide a uniform abstraction over both sparse-tensor
/// types (i.e., `RankedTensorType` with `SparseTensorEncodingAttr`)
/// and dense-tensor types (i.e., `RankedTensorType` without an encoding).
/// That is, we want to manipulate dense-tensor types using the same API
/// that we use for manipulating sparse-tensor types; both to keep the
/// "dimension" vs "level" distinction overt, and to avoid needing to
/// handle certain cases specially in the sparse-compiler.
///
/// (3) To provide uniform handling of "defaults".  In particular
/// this means that dense-tensors should always return the same answers
/// as sparse-tensors with a default encoding.  But it additionally means
/// that the answers should be normalized, so that there's no way to
/// distinguish between non-provided data (which is filled in by default)
/// vs explicitly-provided data which equals the defaults.
///
class SparseTensorType {
public:
  // We memoize `lvlRank` and `dim2lvl` to avoid repeating the
  // conditionals throughout the rest of the class.
  SparseTensorType(RankedTensorType rtp)
      : rtp(rtp), enc(getSparseTensorEncoding(rtp)),
        lvlRank(enc ? enc.getLvlRank() : getDimRank()),
        dim2lvl(enc.hasIdDimOrdering() ? AffineMap() : enc.getDimOrdering()) {
    assert((!isIdentity() || getDimRank() == lvlRank) && "Rank mismatch");
  }

  SparseTensorType(ShapedType stp, SparseTensorEncodingAttr enc)
      : SparseTensorType(
            RankedTensorType::get(stp.getShape(), stp.getElementType(), enc)) {}

  /// Constructs a new `SparseTensorType` with the same dimension-shape
  /// and element type, but with the encoding replaced by the given encoding.
  SparseTensorType withEncoding(SparseTensorEncodingAttr newEnc) const {
    return SparseTensorType(rtp, newEnc);
  }

  /// Constructs a new `SparseTensorType` with the same dimension-shape
  /// and element type, but with the encoding replaced by
  /// `getEncoding().withoutOrdering()`.
  SparseTensorType withoutOrdering() const {
    return withEncoding(enc.withoutOrdering());
  }

  /// Allow implicit conversion to `RankedTensorType`, `ShapedType`,
  /// and `Type`.  These are implicit to help alleviate the impedance
  /// mismatch for code that has not been converted to use `SparseTensorType`
  /// directly.  Once more of the sparse compiler has been converted to
  /// using `SparseTensorType`, we may want to make these explicit instead.
  ///
  /// WARNING: This user-defined-conversion method causes overload
  /// ambiguity whenever passing a `SparseTensorType` directly to a
  /// function which is overloaded to accept either `Type` or `TypeRange`.
  /// In particular, this includes `RewriterBase::replaceOpWithNewOp<OpTy>`
  /// and `OpBuilder::create<OpTy>` whenever the `OpTy::build` is overloaded
  /// thus.  This happens because the `TypeRange<T>(T&&)` ctor is implicit
  /// as well, and there's no SFINAE we can add to this method that would
  /// block subsequent application of that ctor.  The only way to fix the
  /// overload ambiguity is to avoid *implicit* conversion at the callsite:
  /// e.g., by using `static_cast` to make the conversion explicit, by
  /// assigning the `SparseTensorType` to a temporary variable of the
  /// desired type, etc.
  //
  // NOTE: We implement this as a single templated user-defined-conversion
  // function to avoid ambiguity problems when the desired result is `Type`
  // (since both `RankedTensorType` and `ShapedType` can be implicitly
  // converted to `Type`).
  template <typename T, typename = std::enable_if_t<
                            std::is_convertible_v<RankedTensorType, T>>>
  /*implicit*/ operator T() const {
    return rtp;
  }

  /// Explicitly convert to `RankedTensorType`.  This method is
  /// a convenience for resolving overload-ambiguity issues with
  /// implicit conversion.
  RankedTensorType getRankedTensorType() const { return rtp; }

  MLIRContext *getContext() const { return rtp.getContext(); }

  Type getElementType() const { return rtp.getElementType(); }

  /// Returns the encoding (or the null-attribute for dense-tensors).
  SparseTensorEncodingAttr getEncoding() const { return enc; }

  /// Returns true for tensors which have an encoding, and false for
  /// those which do not.  Therefore tensors with an all-dense encoding
  /// return true.
  bool hasEncoding() const { return static_cast<bool>(enc); }

  /// Returns true for tensors where every level is dense.
  /// (This is always true for dense-tensors.)
  bool isAllDense() const { return enc.isAllDense(); }

  /// Returns true for tensors where every level is ordered.
  /// (This is always true for dense-tensors.)
  bool isAllOrdered() const { return enc.isAllOrdered(); }

  /// Returns true if the dimToLvl mapping is the identity.
  /// (This is always true for dense-tensors.)
  bool isIdentity() const { return !dim2lvl; }

  /// Returns the dimToLvl mapping (or the null-map for the identity).
  AffineMap getDimToLvlMap() const { return dim2lvl; }

  /// Returns the dimToLvl mapping, where the identity map is expanded out
  /// into a full `AffineMap`.  This method is provided as a convenience,
  /// but for most purposes other methods (`isIdentity`, `getDimToLvlMap`,
  /// etc) will be more helpful.
  AffineMap getExpandedDimToLvlMap() const {
    return dim2lvl
               ? dim2lvl
               : AffineMap::getMultiDimIdentityMap(getDimRank(), getContext());
  }

  /// Returns the dimension-rank.
  Dimension getDimRank() const { return rtp.getRank(); }

  /// Returns the level-rank.
  Level getLvlRank() const { return lvlRank; }

  /// Returns the dimension-shape.
  ArrayRef<DynSize> getDimShape() const { return rtp.getShape(); }

  /// Safely looks up the requested dimension-DynSize.  If you intend
  /// to check the result with `ShapedType::isDynamic`, then see the
  /// `getStaticDimSize` method instead.
  DynSize getDynamicDimSize(Dimension d) const {
    assert(d < getDimRank() && "Dimension is out of bounds");
    return getDimShape()[d];
  }

  /// Safely looks up the requested dimension-size, mapping dynamic
  /// sizes to `std::nullopt`.
  std::optional<StaticSize> getStaticDimSize(Dimension d) const {
    const DynSize sh = getDynamicDimSize(d);
    return ShapedType::isDynamic(sh) ? std::nullopt
                                     : std::optional<StaticSize>(sh);
  }

  /// Returns true if no dimension has dynamic size.
  bool hasStaticDimShape() const { return rtp.hasStaticShape(); }

  /// Returns true if any dimension has dynamic size.
  bool hasDynamicDimShape() const { return !hasStaticDimShape(); }

  /// Returns true if the given dimension has dynamic size.  If you
  /// intend to call `getDynamicDimSize` based on the result, then see
  /// the `getStaticDimSize` method instead.
  bool isDynamicDim(Dimension d) const {
    // We don't use `rtp.isDynamicDim(d)` because we want the
    // OOB error message to be consistent with `getDynamicDimSize`.
    return ShapedType::isDynamic(getDynamicDimSize(d));
  }

  /// Returns the number of dimensions which have dynamic sizes.
  /// The return type is `int64_t` to maintain consistency with
  /// `ShapedType::Trait<T>::getNumDynamicDims`.
  int64_t getNumDynamicDims() const { return rtp.getNumDynamicDims(); }

  DimLevelType getLvlType(Level l) const {
    // This OOB check is for dense-tensors, since this class knows
    // their lvlRank (whereas STEA::getLvlType will/can only check
    // OOB for sparse-tensors).
    assert(l < lvlRank && "Level out of bounds");
    return enc.getLvlType(l);
  }

  // We can't just delegate these, since we want to use this class's
  // `getLvlType` method instead of STEA's.
  bool isDenseLvl(Level l) const { return isDenseDLT(getLvlType(l)); }
  bool isCompressedLvl(Level l) const { return isCompressedDLT(getLvlType(l)); }
  bool isSingletonLvl(Level l) const { return isSingletonDLT(getLvlType(l)); }
  bool isOrderedLvl(Level l) const { return isOrderedDLT(getLvlType(l)); }
  bool isUniqueLvl(Level l) const { return isUniqueDLT(getLvlType(l)); }

  /// Returns the index-overhead bitwidth, defaulting to zero.
  unsigned getIndexBitWidth() const { return enc ? enc.getIndexBitWidth() : 0; }

  /// Returns the pointer-overhead bitwidth, defaulting to zero.
  unsigned getPointerBitWidth() const {
    return enc ? enc.getPointerBitWidth() : 0;
  }

private:
  // These two must be const, to ensure coherence of the memoized fields.
  const RankedTensorType rtp;
  const SparseTensorEncodingAttr enc;
  // Memoized to avoid frequent redundant conditionals.
  const Level lvlRank;
  const AffineMap dim2lvl;
};

/// Convenience method to abbreviate wrapping `getRankedTensorType`.
template <typename T>
inline SparseTensorType getSparseTensorType(T t) {
  return SparseTensorType(getRankedTensorType(t));
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORTYPE_H_
