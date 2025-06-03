//===- VectorLinearize.h - Vector linearization patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORLINEARIZE_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORLINEARIZE_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace vector {

/// Initialize `typeConverter` with source and target materialization logic
/// using shape_casts to/from 1D vectors.
void initializeForVectorLinearize(TypeConverter &typeConverter);

/// This enum controls the patterns used for linearization of insert,
/// insert_strided_slice, extract, and extract_strided_slice operations.
enum class InsertExtractLinearizePreference {

  /// The lowerings are
  ///   insert,  insert_strided_slice  -> 1D shuffle
  ///   extract, extract_strided_slice -> 1D shuffle
  ///
  /// Even 1D insert_strided_slice and extract_strided_slice are converted to 1D
  /// shuffles. Insert and extract ops on scalar elements are not converted to
  /// 1D shuffles.
  Shuffle = 0,

  /// The preferred lowerings are
  ///   insert,  insert_strided_slice  -> 1D insert_strided_slice
  ///   extract, extract_strided_slice -> 1D extract_strided_slice
  ///
  /// When these lowerings are not possible because the slices are not
  /// contiguous, 1D shuffles are used.
  Strided
};

/// Initialize `conversionTarget`, and `patterns` for linearization. Here
/// linearization means converting a single operation with 1+ vector
/// operand/result of rank>1, into a new single operation whose vector operands
/// and results are all of rank<=1.
///
/// This function initializes `conversionTarget` with the set of operations that
/// are illegal and consequently must be converted to a linearized form. It
/// also populates the set of patterns that can be run to convert illegal
/// operations, and what priority/benefit they have. The patterns and legality
/// rules depend on `preference`, which controls the benefit associated to the
/// patterns based on whether 1D shuffles or 1D strided ops are preferred.
///
/// Note: the set of legal operations can be extended by a user if, for example,
/// certain rank>1 vectors are considered valid, by adding additional
/// dynamically legal ops to `conversionTarget`.
///
/// Further note: the choice to use a dialect conversion design for
/// linearization is to make it easy to reuse generic structural type
/// conversions for linearizing scf/cf/func operations
void populateForFullVectorLinearize(
    const TypeConverter &, ConversionTarget &conversionTarget,
    RewritePatternSet &patterns,
    InsertExtractLinearizePreference preference =
        InsertExtractLinearizePreference::Shuffle);

enum class LinearizePattern {

  /// BEFORE
  /// %1 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
  ///
  /// AFTER
  /// %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  /// %1 = vector.shape_cast %0 : vector<4xf32> to vector<2x2xf32>
  LinearizeConstantLike = 0,

  /// BEFORE
  /// %2 = math.sin %arg0 : vector<2x2xf32>
  ///
  /// AFTER
  /// %0 = vector.shape_cast %arg0 : vector<2x2xf32> to vector<4xf32>
  /// %1 = math.sin %0 : vector<4xf32>
  /// %2 = vector.shape_cast %1 : vector<4xf32> to vector<2x2xf32>
  LinearizeVectorizable,

  /// BEFORE
  /// %2 = vector.bitcast %arg0 : vector<4x4xf32> to vector<4x8xf16>
  ///
  /// AFTER
  /// %0 = vector.shape_cast %arg0 : vector<4x4xf32> to vector<16xf32>
  /// %1 = vector.bitcast %0 : vector<16xf32> to vector<32xf16>
  /// %2 = vector.shape_cast %1 : vector<32xf16> to vector<4x8xf16>
  LinearizeVectorBitCast,

  /// This pattern currently only supports 2D masks with a unit outer
  /// dimension.
  ///
  /// BEFORE
  /// %mask_2d = vector.create_mask %arg0, %arg1 : vector<1x4xi1>
  ///
  /// AFTER
  /// [...]
  /// %mask_1d= vector.create_mask %mul : vector<4xi1>
  /// %mask_2d = vector.shape_cast %mask : vector<4xi1> to vector<1x4xi1>
  ///
  /// where `%mul` is a function of `%arg0` and `%arg1`.
  LinearizeVectorCreateMask,

  /// BEFORE
  /// %shuffle_3d = vector.shuffle %v1_3d, %v2_3d [ shuffle_indices ]
  ///
  /// AFTER
  /// %v1_1d = vector.shape_cast %v1_3d : [...]
  /// %v2_1d = vector.shape_cast %v2_3d : [...]
  /// %shuffle_1d = vector.shuffle %v1_1d, %v2_1d [ shuffle_indices_1d ]
  /// %shuffle_3d = vector.shape_cast %shuffle_1d :  [...]
  ///
  /// Where `shuffle_indices_1d` are computed by expanding `shuffle_indices`.
  LinearizeVectorShuffle,

  /// BEFORE
  /// %1 = vector.splat %value : vector<4x4xf32>
  ///
  /// AFTER
  /// %0 = vector.splat %value : vector<16xf32>
  /// %1 = vector.shape_cast %0 : vector<16xf32> to vector<4x4xf32>
  LinearizeVectorSplat,

  /// Reduce the rank of a vector.extract_strided_slice to the lowest rank
  /// possible. For extract_strided_slice ops that slice contiguous elements,
  /// the reduced-rank op is 1D, otherwise it is higher dimensional.
  ///
  /// BEFORE
  /// %2 = vector.extract_strided_slice %arg0 {
  ///    offsets = [1, 0, 1, 0],
  ///      sizes = [1, 2, 1, 2],
  ///    strides = [1, 1, 1, 1]} : vector<2x2x2x2xi8> to vector<1x2x1x2xi8>
  ///
  /// AFTER
  /// %0 = vector.shape_cast %arg0 : vector<2x2x2x2xi8> to vector<4x4xi8>
  /// %1 = vector.extract_strided_slice %0 {
  ///    offsets = [2, 2],
  ///      sizes = [2, 2],
  ///    strides = [1, 1]} : vector<4x4xi8> to vector<2x2xi8>
  /// %2 = vector.shape_cast %1 : vector<2x2xi8> to vector<1x2x1x2xi8>
  RankReduceExtractStridedSlice,

  /// Similar to RankReduceExtractStridedSlice, but both the operands have
  /// their rank reduced.
  ///
  /// BEFORE
  /// %3 = vector.insert_strided_slice %arg1, %arg0 {[...]}
  ///                vector<1x2x1x2xi8> into vector<2x2x2x2xi8>
  ///
  /// AFTER
  /// %0 = vector.shape_cast %arg0 : vector<2x2x2x2xi8> to vector<4x4xi8>
  /// %1 = vector.shape_cast %arg1 : vector<1x2x1x2xi8> to vector<2x2xi8>
  /// %2 = vector.insert_strided_slice %1, %0 {[...]}
  /// %3 = vector.shape_cast %2 : vector<4x4xi8> to vector<2x2x2x2xi8>
  RankReduceInsertStridedSlice,

  /// BEFORE
  /// %extract = vector.extract %src [ position ]
  ///
  /// AFTER
  /// %src_1d = vector.shape_cast %src : [...]
  /// %out_1d = vector.shuffle %source_1d, %source_1d [ shuffle_indices ]
  /// %out_nd = vector.shape_cast %out_1d : [...]
  ///
  /// `shuffle_indices` is computed from `position`.
  VectorExtractToRankOneShuffle,

  /// BEFORE
  /// %out_nd = vector.extract_strided_slice %source_nd
  ///         { offsets = [..], strides = [..], sizes = [..] }
  ///
  /// AFTER
  /// %source_1d = vector.shape_cast %source_nd [...]
  /// %out_1d    = vector.shuffle %source_1d, %source_1d [ shuffle_indices_1d ]
  /// %out_nd    = vector.shape_cast %out_1d [...]
  ///
  /// `shuffle_indices_1d` is computed using the offsets and sizes of the
  /// original vector.extract_strided_slice operation.
  VectorExtractStridedSliceToRankOneShuffle,

  /// BEFORE
  /// %1 = vector.extract %arg0[1, 2] : vector<2x1xi8> from vector<4x3x2x1xi8>
  ///
  /// AFTER
  /// %0 = vector.shape_cast %arg0 : vector<4x3x2x1xi8> to vector<24xi8>
  /// %1 = vector.extract_strided_slice %0 {offsets = [10], sizes = [2] [...]
  /// %2 = vector.shape_cast %1 : vector<2xi8> to vector<2x1xi8>
  VectorExtractToRankOneStrided,

  /// BEFORE
  /// %insert = vector.insert %src %dst [ position ]
  ///
  /// AFTER
  /// %src_1d = vector.shape_cast %src : [...]
  /// %dst_1d = vector.shape_cast %dst : [...]
  /// %out_1d = vector.shuffle %dst_1d, %src_1d [ shuffle_indices ]
  /// %out_nd = vector.shape_cast %out_1d : [...]
  ///
  /// `shuffle_indices` is computed from `position`.
  VectorInsertToRankOneShuffle,

  /// This pattern converts a vector.insert_strided_slice operation into a
  /// vector.shuffle operation that has rank-1 (linearized) operands and result.
  ///
  /// BEFORE
  /// %0 = vector.insert_strided_slice %to_store, %into
  ///             {offsets = [1, 0, 0, 0], strides = [1, 1]}
  ///                  : vector<2x2xi8> into vector<2x1x3x2xi8>
  /// AFTER
  /// %to_store_1d
  ///          = vector.shape_cast %to_store : vector<2x2xi8> to vector<4xi8>
  /// %into_1d = vector.shape_cast %into : vector<2x1x3x2xi8> to vector<12xi8>
  /// %out_1d  = vector.shuffle %into_1d, %to_store_1d [ shuffle_indices_1d ]
  /// %out_nd  = vector.shape_cast %out_1d : vector<12xi8> to vector<2x1x3x2xi8>
  ///
  /// where shuffle_indices_1d in this case is
  ///     [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 10, 11].
  ///                        ^^^^^^^^^^^^^^
  ///                          to_store_1d
  VectorInsertStridedSliceToRankOneShuffle,

  /// Similar to VectorExtractToRankOneStrided, but for insert_strided_slice.
  VectorInsertToRankOneStrided,

  /// The number of patterns in this enum.
  N
};

/// This class contains functions to control the set of linearization patterns
/// to include for the conversion, and their priority.
struct VectorLinearizePatterns {

public:
  /// By default all patterns are enabled and have benefit 1.
  VectorLinearizePatterns() {
    enabled.fill(true);
    benefits.fill(PatternBenefit(1));
  }

  /// Add the patterns enabled for the conversion to `patterns`.
  void addToPatternSet(const TypeConverter &,
                       RewritePatternSet &patterns) const;

  VectorLinearizePatterns &enable(LinearizePattern id, bool e = true) {
    enabled[static_cast<unsigned>(id)] = e;
    return *this;
  }

  VectorLinearizePatterns &enableAll(bool e = true) {
    enabled.fill(e);
    return *this;
  }

  bool isEnabled(LinearizePattern id) const {
    return enabled[static_cast<unsigned>(id)];
  }

  PatternBenefit getBenefit(LinearizePattern id) const {
    return benefits[static_cast<unsigned>(id)];
  }

  VectorLinearizePatterns &setBenefit(LinearizePattern id,
                                      PatternBenefit benefit) {
    getBenefitRef(id) = benefit;
    return *this;
  }

  VectorLinearizePatterns &incrementBenefit(LinearizePattern id,
                                            unsigned inc = 1) {
    getBenefitRef(id) = getBenefit(id).getBenefit() + 1;
    return *this;
  }

private:
  std::array<bool, static_cast<unsigned>(LinearizePattern::N)> enabled;
  std::array<PatternBenefit, static_cast<unsigned>(LinearizePattern::N)>
      benefits;

  PatternBenefit &getBenefitRef(LinearizePattern id) {
    unsigned idInt = static_cast<unsigned>(id);
    assert(idInt < static_cast<unsigned>(LinearizePattern::N) &&
           "invalid linearization pattern id");
    return benefits[idInt];
  }
};

/// Consider inserting a vector of shape `small` into a vector of shape `large`,
/// at position `offsets`: this function enumeratates all the indices in `large`
/// that are written to. The enumeration is with row-major ordering.
///
/// Example: insert a 1x2 vector into a 4x5 vector at position (1,3). The 2
/// positions written to are (1,3) and (1,4), which have linearized indices 8
/// and 9. So [8,9] is returned.
///
/// The length of the returned vector is equal to the number of elements in
/// the shape `small` (i.e. the product of dimensions of `small`).
SmallVector<int64_t> getStridedSliceInsertionIndices(ArrayRef<int64_t> small,
                                                     ArrayRef<int64_t> large,
                                                     ArrayRef<int64_t> offsets);

/// Return the strided slice with the lowest rank that is equivalent to the
/// strided slice of `small` from `large`, starting at `offsets`. The result is
/// a tuple of three vectors:
///
/// 0) The shape of the new small vector.
/// 1) The shape of the new large vector.
/// 2) The offsets of the new large vector.
///
/// Example 1 (contiguous slices can always be represented in 1-D).
///
/// Input:
///  small  = (1, 3, 4)
///  large  = (3, 3, 4)
///  offset = (2, 3, 4)
///
/// Output:
///  small  = (12)
///  large  = (36)
///  offset = (24)
///
/// Example 2 (a non-contiguous slice)
///
/// Input:
///  small  =    (2, 2, 1, 2)
///  large  = (2, 2, 2, 2, 2)
///  offset = (1, 1, 0, 1)
///
///
/// Output:
///  small  =  (4, 2)
///  large  =  (8, 4)
///  offset = (24, 2)
std::array<SmallVector<int64_t>, 3>
getCollapsedStridedSliceShape(ArrayRef<int64_t> small, ArrayRef<int64_t> large,
                              ArrayRef<int64_t> offsets);

std::optional<std::array<SmallVector<int64_t>, 3>>
getCollapsedExtractStridedSliceShape(vector::ExtractStridedSliceOp extractOp);

std::optional<std::array<SmallVector<int64_t>, 3>>
getCollapsedInsertStridedSliceShape(vector::InsertStridedSliceOp insertOp);

} // namespace vector
} // namespace mlir

#endif
