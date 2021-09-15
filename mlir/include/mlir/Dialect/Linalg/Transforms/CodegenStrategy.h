//===- CodegenStrategy.h - Linalg programmable codegen strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {

class FuncOp;

namespace linalg {

/// Abstract Transformation class applied in a sequence that also handles state
/// through markers.
struct Transformation {
  explicit Transformation(linalg::LinalgTransformationFilter::FilterFunction f)
      : filter(f) {}
  virtual ~Transformation() = default;
  virtual RewritePatternSet
  buildRewritePatterns(MLIRContext *context,
                       linalg::LinalgTransformationFilter m) = 0;
  linalg::LinalgTransformationFilter::FilterFunction filter = nullptr;
};

/// SFINAE: Enqueue helper for ConcreteOpType that have a `getOperationName`.
template <template <typename> class PatternType, typename ConcreteOpType,
          typename OptionsType,
          typename = std::enable_if_t<std::is_member_function_pointer<
              decltype(&ConcreteOpType::getOperationName)>::value>>
void sfinae_enqueue(RewritePatternSet &patternList, OptionsType options,
                    StringRef opName, linalg::LinalgTransformationFilter m) {
  assert(opName == ConcreteOpType::getOperationName() &&
         "explicit name must match ConcreteOpType::getOperationName");
  patternList.add<PatternType<ConcreteOpType>>(patternList.getContext(),
                                               options, m);
}

/// SFINAE: Enqueue helper for OpType that do not have a `getOperationName`
/// (e.g. LinalgOp, other interfaces, Operation*).
template <template <typename> class PatternType, typename OpType,
          typename OptionsType>
void sfinae_enqueue(RewritePatternSet &patternList, OptionsType options,
                    StringRef opName, linalg::LinalgTransformationFilter m) {
  assert(!opName.empty() && "opName must not be empty");
  patternList.add<PatternType<OpType>>(opName, patternList.getContext(),
                                       options, m);
}

template <typename PatternType, typename OpType, typename OptionsType>
void enqueue(RewritePatternSet &patternList, OptionsType options,
             StringRef opName, linalg::LinalgTransformationFilter m) {
  if (!opName.empty())
    patternList.add<PatternType>(opName, patternList.getContext(), options, m);
  else
    patternList.add<PatternType>(patternList.getContext(),
                                 m.addOpFilter<OpType>(), options);
}

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Tile<LinalgOpType>`with the appropriate `options`.
template <typename LinalgOpType>
struct Tile : public Transformation {
  explicit Tile(linalg::LinalgTilingOptions options,
                linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(LinalgOpType::getOperationName()),
        options(options) {}

  Tile(StringRef name, linalg::LinalgTilingOptions options,
       linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  RewritePatternSet
  buildRewritePatterns(MLIRContext *context,
                       linalg::LinalgTransformationFilter m) override {
    RewritePatternSet tilingPatterns(context);
    sfinae_enqueue<linalg::LinalgTilingPattern, LinalgOpType>(
        tilingPatterns, options, opName, m);
    return tilingPatterns;
  }

private:
  std::string opName;
  linalg::LinalgTilingOptions options;
};

/// Promotion transformation enqueues a particular stage-1 pattern for
/// `Promote<LinalgOpType>`with the appropriate `options`.
template <typename LinalgOpType>
struct Promote : public Transformation {
  explicit Promote(
      linalg::LinalgPromotionOptions options,
      linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(LinalgOpType::getOperationName()),
        options(options) {}

  Promote(StringRef name, linalg::LinalgPromotionOptions options,
          linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  RewritePatternSet
  buildRewritePatterns(MLIRContext *context,
                       linalg::LinalgTransformationFilter m) override {
    RewritePatternSet promotionPatterns(context);
    sfinae_enqueue<linalg::LinalgPromotionPattern, LinalgOpType>(
        promotionPatterns, options, opName, m);
    return promotionPatterns;
  }

private:
  std::string opName;
  linalg::LinalgPromotionOptions options;
};

/// Vectorization transformation enqueues a particular stage-1 pattern for
/// `LinalgVectorizationPattern<LinalgOpType>` as well as copy to vector
/// transfer rewrite forwarding patterns.
template <typename LinalgOpType = LinalgOp>
struct Vectorize : public Transformation {
  explicit Vectorize(
      linalg::LinalgVectorizationOptions options,
      linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(), options(options) {}

  Vectorize(StringRef name, linalg::LinalgVectorizationOptions options,
            linalg::LinalgTransformationFilter::FilterFunction f = nullptr)
      : Transformation(f), opName(name), options(options) {}

  RewritePatternSet
  buildRewritePatterns(MLIRContext *context,
                       linalg::LinalgTransformationFilter m) override {
    RewritePatternSet vectorizationPatterns(context);
    enqueue<linalg::LinalgVectorizationPattern, LinalgOpType>(
        vectorizationPatterns, options, opName, m);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        context, /*benefit=*/2);
    return vectorizationPatterns;
  }

private:
  std::string opName;
  linalg::LinalgVectorizationOptions options;
};

/// Options to control the application of late transformations.
struct LateCodegenStrategyOptions {
  /// Hoisting transformations are always deemed beneficial and must disabled
  /// explicitly.
  bool enableLICM = true;
  bool enableHoistRedundantVectorTransfers = true;
  bool enableHoistRedundantVectorTransfersOnTensor = true;
  /// Vector lowering operations may result in surprising behavior when
  /// composing multiple codegen strategies and must be enabled explicitly.
  bool enableVectorTransferPartialRewrite = false;
  bool enableVectorContractLowering = false;
  bool enableVectorToSCFConversion = false;
};

/// Codegen strategy controls how a Linalg op is progressively lowered.
/// The application uses a 3-level staged patterns strategy which allows
/// ordering transformations by using the Linalg `applyStagedPatterns`
/// function, where:
///   1. The first stage consists of the successive `tile`, `promote` and
///   `vectorize` patterns, applied sequentially.
///   2. The second stage consists of common local canonicalization patterns
///   that are applied eagerly after each stage-1 pattern.
///   3. the third stage consists of more global transformation, also applied
///   eagerly, after all stage-2 patterns. Such more global transformations
struct CodegenStrategy {
  /// Append a pattern to add a level of tiling for `LinalgOpType` with tiling
  /// `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  tile(linalg::LinalgTilingOptions options,
       linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Tile<LinalgOpType>>(options, f));
    return *this;
  }
  /// Append a pattern to add a level of tiling for `LinalgOpType` with tiling
  /// `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  tile(StringRef opName, linalg::LinalgTilingOptions options,
       linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Tile<LinalgOpType>>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of tiling for
  /// `LinalgOpType` with tiling `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  tileIf(bool b, linalg::LinalgTilingOptions options,
         linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tile<LinalgOpType>(options) : *this;
  }
  /// Conditionally append a pattern to add a level of tiling for
  /// `LinalgOpType` with tiling `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  tileIf(bool b, StringRef opName, linalg::LinalgTilingOptions options,
         linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? tile<LinalgOpType>(opName, options) : *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  promote(linalg::LinalgPromotionOptions options,
          linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Promote<LinalgOpType>>(options, f));
    return *this;
  }
  /// Append a pattern to add a level of promotion for `LinalgOpType` with
  /// promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  promote(StringRef opName, linalg::LinalgPromotionOptions options,
          linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Promote<LinalgOpType>>(opName, options, f));
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  promoteIf(bool b, StringRef opName, linalg::LinalgPromotionOptions options,
            linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? promote<LinalgOpType>(opName, options, f) : *this;
    return *this;
  }
  /// Conditionally append a pattern to add a level of promotion for
  /// `LinalgOpType` with promotion `options`.
  template <typename LinalgOpType>
  CodegenStrategy &
  promoteIf(bool b, linalg::LinalgPromotionOptions options,
            linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? promote<LinalgOpType>(options, f) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  template <typename LinalgOpType>
  CodegenStrategy &
  vectorize(linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    transformationSequence.emplace_back(
        std::make_unique<Vectorize<LinalgOpType>>(
            linalg::LinalgVectorizationOptions(), f));
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  template <typename LinalgOpType>
  CodegenStrategy &
  vectorizeIf(bool b,
              linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? vectorize<LinalgOpType>(f) : *this;
    return *this;
  }
  /// Append a pattern to rewrite `LinalgOpType` as a vector operation.
  CodegenStrategy &
  vectorize(StringRef opName,
            linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    assert(!opName.empty() && "expected an op name");
    transformationSequence.emplace_back(std::make_unique<Vectorize<LinalgOp>>(
        opName, linalg::LinalgVectorizationOptions(), f));
    return *this;
  }
  /// Conditionally append a pattern to rewrite `LinalgOpType` as a vector
  /// operation.
  CodegenStrategy &
  vectorizeIf(bool b, StringRef opName,
              linalg::LinalgTransformationFilter::FilterFunction f = nullptr) {
    return b ? vectorize(opName, f) : *this;
    return *this;
  }
  /// Configure the post staged-patterns late vector transformations.
  CodegenStrategy &
  setVectorTransformsOptions(vector::VectorTransformsOptions options) {
    vectorTransformsOptions = options;
    return *this;
  }
  /// Configure the post staged-patterns late vector.transfer to scf
  /// conversion.
  CodegenStrategy &
  setVectorTransferToSCFOptions(VectorTransferToSCFOptions options) {
    vectorToSCFOptions = options;
    return *this;
  }
  ///
  /// Configure the application of late transformations.
  ///
  CodegenStrategy &setEnableLICM(bool val) {
    this->lateCodegenStrategyOptions.enableLICM = val;
    return *this;
  }
  CodegenStrategy &setEnableHoistRedundantVectorTransfers(bool val) {
    this->lateCodegenStrategyOptions.enableHoistRedundantVectorTransfers = val;
    return *this;
  }
  CodegenStrategy &setEnableHoistRedundantVectorTransfersOnTensor(bool val) {
    this->lateCodegenStrategyOptions
        .enableHoistRedundantVectorTransfersOnTensor = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorTransferPartialRewrite(bool val) {
    this->lateCodegenStrategyOptions.enableVectorTransferPartialRewrite = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorContractLowering(bool val) {
    this->lateCodegenStrategyOptions.enableVectorContractLowering = val;
    return *this;
  }
  CodegenStrategy &setEnableVectorToSCFConversion(bool val) {
    this->lateCodegenStrategyOptions.enableVectorToSCFConversion = val;
    return *this;
  }

  /// Apply the transformation patterns in sequence with cleanup
  /// transformations interleaved.
  void transform(FuncOp func) const;

private:
  LogicalResult postPatternTransforms(Operation *func) const;

  vector::VectorTransformsOptions vectorTransformsOptions;
  VectorTransferToSCFOptions vectorToSCFOptions;
  SmallVector<std::unique_ptr<Transformation>, 4> transformationSequence;
  LateCodegenStrategyOptions lateCodegenStrategyOptions;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_CODEGENSTRATEGY_H_
