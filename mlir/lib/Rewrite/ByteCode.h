//===- ByteCode.h - Pattern byte-code interpreter ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a byte-code and interpreter for pattern rewrites in MLIR.
// The byte-code is constructed from the PDL Interpreter dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_BYTECODE_H_
#define MLIR_REWRITE_BYTECODE_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pdl_interp {
class RecordMatchOp;
} // end namespace pdl_interp

namespace detail {
class PDLByteCode;

/// Use generic bytecode types. ByteCodeField refers to the actual bytecode
/// entries (set to uint8_t for "byte" bytecode). ByteCodeAddr refers to size of
/// indices into the bytecode. Correctness is checked with static asserts.
using ByteCodeField = uint16_t;
using ByteCodeAddr = uint32_t;

//===----------------------------------------------------------------------===//
// PDLByteCodePattern
//===----------------------------------------------------------------------===//

/// All of the data pertaining to a specific pattern within the bytecode.
class PDLByteCodePattern : public Pattern {
public:
  static PDLByteCodePattern create(pdl_interp::RecordMatchOp matchOp,
                                   ByteCodeAddr rewriterAddr);

  /// Return the bytecode address of the rewriter for this pattern.
  ByteCodeAddr getRewriterAddr() const { return rewriterAddr; }

private:
  template <typename... Args>
  PDLByteCodePattern(ByteCodeAddr rewriterAddr, Args &&...patternArgs)
      : Pattern(std::forward<Args>(patternArgs)...),
        rewriterAddr(rewriterAddr) {}

  /// The address of the rewriter for this pattern.
  ByteCodeAddr rewriterAddr;
};

//===----------------------------------------------------------------------===//
// PDLByteCodeMutableState
//===----------------------------------------------------------------------===//

/// This class contains the mutable state of a bytecode instance. This allows
/// for a bytecode instance to be cached and reused across various different
/// threads/drivers.
class PDLByteCodeMutableState {
public:
  /// Initialize the state from a bytecode instance.
  void initialize(PDLByteCode &bytecode);

  /// Set the new benefit for a bytecode pattern. The `patternIndex` corresponds
  /// to the position of the pattern within the range returned by
  /// `PDLByteCode::getPatterns`.
  void updatePatternBenefit(unsigned patternIndex, PatternBenefit benefit);

private:
  /// Allow access to data fields.
  friend class PDLByteCode;

  /// The mutable block of memory used during the matching and rewriting phases
  /// of the bytecode.
  std::vector<const void *> memory;

  /// The up-to-date benefits of the patterns held by the bytecode. The order
  /// of this array corresponds 1-1 with the array of patterns in `PDLByteCode`.
  std::vector<PatternBenefit> currentPatternBenefits;
};

//===----------------------------------------------------------------------===//
// PDLByteCode
//===----------------------------------------------------------------------===//

/// The bytecode class is also the interpreter. Contains the bytecode itself,
/// the static info, addresses of the rewriter functions, the interpreter
/// memory buffer, and the execution context.
class PDLByteCode {
public:
  /// Each successful match returns a MatchResult, which contains information
  /// necessary to execute the rewriter and indicates the originating pattern.
  struct MatchResult {
    MatchResult(Location loc, const PDLByteCodePattern &pattern,
                PatternBenefit benefit)
        : location(loc), pattern(&pattern), benefit(benefit) {}

    /// The location of operations to be replaced.
    Location location;
    /// Memory values defined in the matcher that are passed to the rewriter.
    SmallVector<const void *, 4> values;
    /// The originating pattern that was matched. This is always non-null, but
    /// represented with a pointer to allow for assignment.
    const PDLByteCodePattern *pattern;
    /// The current benefit of the pattern that was matched.
    PatternBenefit benefit;
  };

  /// Create a ByteCode instance from the given module containing operations in
  /// the PDL interpreter dialect.
  PDLByteCode(ModuleOp module,
              llvm::StringMap<PDLConstraintFunction> constraintFns,
              llvm::StringMap<PDLRewriteFunction> rewriteFns);

  /// Return the patterns held by the bytecode.
  ArrayRef<PDLByteCodePattern> getPatterns() const { return patterns; }

  /// Initialize the given state such that it can be used to execute the current
  /// bytecode.
  void initializeMutableState(PDLByteCodeMutableState &state) const;

  /// Run the pattern matcher on the given root operation, collecting the
  /// matched patterns in `matches`.
  void match(Operation *op, PatternRewriter &rewriter,
             SmallVectorImpl<MatchResult> &matches,
             PDLByteCodeMutableState &state) const;

  /// Run the rewriter of the given pattern that was previously matched in
  /// `match`.
  void rewrite(PatternRewriter &rewriter, const MatchResult &match,
               PDLByteCodeMutableState &state) const;

private:
  /// Execute the given byte code starting at the provided instruction `inst`.
  /// `matches` is an optional field provided when this function is executed in
  /// a matching context.
  void executeByteCode(const ByteCodeField *inst, PatternRewriter &rewriter,
                       PDLByteCodeMutableState &state,
                       SmallVectorImpl<MatchResult> *matches) const;

  /// A vector containing pointers to uniqued data. The storage is intentionally
  /// opaque such that we can store a wide range of data types. The types of
  /// data stored here include:
  ///  * Attribute, Identifier, OperationName, Type
  std::vector<const void *> uniquedData;

  /// A vector containing the generated bytecode for the matcher.
  SmallVector<ByteCodeField, 64> matcherByteCode;

  /// A vector containing the generated bytecode for all of the rewriters.
  SmallVector<ByteCodeField, 64> rewriterByteCode;

  /// The set of patterns contained within the bytecode.
  SmallVector<PDLByteCodePattern, 32> patterns;

  /// A set of user defined functions invoked via PDL.
  std::vector<PDLConstraintFunction> constraintFunctions;
  std::vector<PDLRewriteFunction> rewriteFunctions;

  /// The maximum memory index used by a value.
  ByteCodeField maxValueMemoryIndex = 0;
};

} // end namespace detail
} // end namespace mlir

#endif // MLIR_REWRITE_BYTECODE_H_
