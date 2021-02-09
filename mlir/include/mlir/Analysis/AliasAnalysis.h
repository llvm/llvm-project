//===- AliasAnalysis.h - Alias Analysis in MLIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and analyses for performing alias queries
// and related memory queries in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_ALIASANALYSIS_H_
#define MLIR_ANALYSIS_ALIASANALYSIS_H_

#include "mlir/IR/Operation.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AliasResult
//===----------------------------------------------------------------------===//

/// The possible results of an alias query.
class AliasResult {
public:
  enum Kind {
    /// The two locations do not alias at all.
    ///
    /// This value is arranged to convert to false, while all other values
    /// convert to true. This allows a boolean context to convert the result to
    /// a binary flag indicating whether there is the possibility of aliasing.
    NoAlias = 0,
    /// The two locations may or may not alias. This is the least precise
    /// result.
    MayAlias,
    /// The two locations alias, but only due to a partial overlap.
    PartialAlias,
    /// The two locations precisely alias each other.
    MustAlias,
  };

  AliasResult(Kind kind) : kind(kind) {}
  bool operator==(const AliasResult &other) const { return kind == other.kind; }
  bool operator!=(const AliasResult &other) const { return !(*this == other); }

  /// Allow conversion to bool to signal if there is an aliasing or not.
  explicit operator bool() const { return kind != NoAlias; }

  /// Merge this alias result with `other` and return a new result that
  /// represents the conservative merge of both results. If the results
  /// represent a known alias, the stronger alias is chosen (i.e.
  /// Partial+Must=Must). If the two results are conflicting, MayAlias is
  /// returned.
  AliasResult merge(AliasResult other) const;

  /// Returns if this result is a partial alias.
  bool isNo() const { return kind == NoAlias; }

  /// Returns if this result is a may alias.
  bool isMay() const { return kind == MayAlias; }

  /// Returns if this result is a must alias.
  bool isMust() const { return kind == MustAlias; }

  /// Returns if this result is a partial alias.
  bool isPartial() const { return kind == PartialAlias; }

  /// Return the internal kind of this alias result.
  Kind getKind() const { return kind; }

private:
  /// The internal kind of the result.
  Kind kind;
};

//===----------------------------------------------------------------------===//
// AliasAnalysisTraits
//===----------------------------------------------------------------------===//

namespace detail {
/// This class contains various internal trait classes used by the main
/// AliasAnalysis class below.
struct AliasAnalysisTraits {
  /// This class represents the `Concept` of an alias analysis implementation.
  /// It is the abstract base class used by the AliasAnalysis class for
  /// querying into derived analysis implementations.
  class Concept {
  public:
    virtual ~Concept() {}

    /// Given two values, return their aliasing behavior.
    virtual AliasResult alias(Value lhs, Value rhs) = 0;
  };

  /// This class represents the `Model` of an alias analysis implementation
  /// `ImplT`. A model is instantiated for each alias analysis implementation
  /// to implement the `Concept` without the need for the derived
  /// implementation to inherit from the `Concept` class.
  template <typename ImplT> class Model final : public Concept {
  public:
    explicit Model(ImplT &&impl) : impl(std::forward<ImplT>(impl)) {}
    ~Model() override = default;

    /// Given two values, return their aliasing behavior.
    AliasResult alias(Value lhs, Value rhs) final {
      return impl.alias(lhs, rhs);
    }

  private:
    ImplT impl;
  };
};
} // end namespace detail

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

/// This class represents the main alias analysis interface in MLIR. It
/// functions as an aggregate of various different alias analysis
/// implementations. This aggregation allows for utilizing the strengths of
/// different alias analysis implementations that either target or have access
/// to different aliasing information. This is especially important for MLIR
/// given the scope of different types of memory models and aliasing behaviors.
/// For users of this analysis that want to perform aliasing queries, see the
/// `Alias Queries` section below for the available methods. For users of this
/// analysis that want to add a new alias analysis implementation to the
/// aggregate, see the `Alias Implementations` section below.
class AliasAnalysis {
  using Concept = detail::AliasAnalysisTraits::Concept;
  template <typename ImplT>
  using Model = detail::AliasAnalysisTraits::Model<ImplT>;

public:
  AliasAnalysis(Operation *op);

  //===--------------------------------------------------------------------===//
  // Alias Implementations
  //===--------------------------------------------------------------------===//

  /// Add a new alias analysis implementation `AnalysisT` to this analysis
  /// aggregate. This allows for users to access this implementation when
  /// performing alias queries. Implementations added here must provide the
  /// following:
  ///   * AnalysisT(AnalysisT &&)
  ///   * AliasResult alias(Value lhs, Value rhs)
  ///     - This method returns an `AliasResult` that corresponds to the
  ///       aliasing behavior between `lhs` and `rhs`.
  template <typename AnalysisT>
  void addAnalysisImplementation(AnalysisT &&analysis) {
    aliasImpls.push_back(
        std::make_unique<Model<AnalysisT>>(std::forward<AnalysisT>(analysis)));
  }

  //===--------------------------------------------------------------------===//
  // Alias Queries
  //===--------------------------------------------------------------------===//

  /// Given two values, return their aliasing behavior.
  AliasResult alias(Value lhs, Value rhs);

private:
  /// A set of internal alias analysis implementations.
  SmallVector<std::unique_ptr<Concept>, 4> aliasImpls;
};

} // end namespace mlir

#endif // MLIR_ANALYSIS_ALIASANALYSIS_H_
