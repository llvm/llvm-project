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

/// Initialize `typeConverter` with source and target materializations that
/// use shape_casts to/from 1D vectors.
void initializeForVectorLinearize(TypeConverter &typeConverter);

/// Initialize `conversionTarget` and `patterns` for linearization. Here
/// linearization means converting a single operation with 1+ vector
/// operand/result of rank>1, into a new single operation whose vector operands
/// and results are all rank<=1.
///
/// This function initializes `conversionTarget` with a definition of which
/// operations are illegal and consequently must be converted to a linearized
/// (legal) form. It also populates `patterns` with the patterns that will be
/// run to convert illegal operations, and what sets what priority/benefit they
/// have.
///
/// Note: the set of legal operations can be extended by a user by adding
/// additional legality rules to `conversionTarget`.
///
/// Further note: the choice to use a dialect conversion design for
/// linearization is to enable reuse of generic structural type conversions for
/// linearizing scf/cf/func operations.
void populateForFullVectorLinearize(const TypeConverter &,
                                    ConversionTarget &conversionTarget,
                                    RewritePatternSet &patterns);

/// The set of patterns available for linearization.
enum class LinearizePattern {

  /// This pattern converts a constant (or poison) vector of rank>1 into a
  /// 1D vector followed by a shape_cast.
  ///
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

  /// This pattern converts the ShuffleOp that works on nD (n > 1)
  /// vectors to a ShuffleOp that works on linearized vectors.
  ///
  /// BEFORE
  /// %shuffle_3d = vector.shuffle %v1_3d, %v2_3d [ shuffle_indices ]
  ///
  /// AFTER
  /// %v1_1d = vector.shape_cast %v1_3d : [...]
  /// %v2_1d = vector.shape_cast %v2_3d : [...]
  /// %shuffle_1d = vector.shuffle %v1_1d, %v2_1d [ shuffle_indices_1d ]
  /// %shuffle_3d = vector.shape_cast %shuffle_1d :  [...]
  ///
  /// Where `shuffle_indices_1d` is computed by expanding `shuffle_indices`.
  LinearizeVectorShuffle,

  /// BEFORE
  /// %1 = vector.splat %value : vector<4x4xf32>
  ///
  /// AFTER
  /// %0 = vector.splat %value : vector<16xf32>
  /// %1 = vector.shape_cast %0 : vector<16xf32> to vector<4x4xf32>
  LinearizeVectorSplat,

  /// BEFORE
  /// %extract = vector.extract %src [ position ]
  ///
  /// AFTER
  /// %src_1d = vector.shape_cast %src : [...]
  /// %out_1d = vector.shuffle %source_1d, %source_1d [ shuffle_indices ]
  /// %out_nd = vector.shape_cast %out_1d : [...]
  ///
  /// `shuffle_indices` is computed from `position` of original extract.
  VectorExtractToRankOneShuffle,

  /// This pattern converts a vector.extract_strided_slice operation into a
  /// vector.shuffle operation that has a rank-1 (linearized) operand and
  /// result.
  ///
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
/// at position `offsets`: this function enumerates all the indices in `large`
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

} // namespace vector
} // namespace mlir

#endif
