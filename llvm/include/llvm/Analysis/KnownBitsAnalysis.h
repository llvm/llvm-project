//===- KnownBitsAnalysis.h - An analysis that caches KnownBits ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Used to cache KnownBits for the entire function, with dataflow-dependent
// invalidation.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_KNOWNBITSANALYSIS_H
#define LLVM_ANALYSIS_KNOWNBITSANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/KnownBits.h"

namespace llvm {
class KnownBitsDataflow;
class Value;
class User;
class Function;
class Operator;
class DataLayout;
class raw_ostream;
struct SimplifyQuery;

class KnownBitsDataflow : protected DenseMap<const Value *, KnownBits> {
  /// The roots are the arguments of the function, and PHI nodes and
  /// Instructions like fptosi in each Basic Block, filtered for integer,
  /// pointer, or vector thereof, types.
  SmallVector<const Value *> Roots;

  SmallVector<const Value *> getLeaves(ArrayRef<const Value *> Roots) const;
  void emplace_all_conflict(const Value *V); // NOLINT

  template <typename RangeT>
  SmallVector<Value *> insert_range(RangeT R); // NOLINT

  void recurseInsertChildren(ArrayRef<Value *> R);
  template <typename RangeT> void initialize(RangeT R);
  void initialize(Function &F);

protected:
  const DataLayout &DL;
  LLVM_ABI_FOR_TEST KnownBits &getKnownBits(const Value *V) {
    return operator[](V);
  }
  LLVM_ABI_FOR_TEST void setAllConflict(const Value *V) {
    assert(contains(V) && "Expected Value in map");
    getKnownBits(V).setAllConflict();
  }
  LLVM_ABI_FOR_TEST void setAllZero(const Value *V) {
    assert(contains(V) && "Expected Value in map");
    getKnownBits(V).setAllZero();
  }
  LLVM_ABI_FOR_TEST void setAllOnes(const Value *V) {
    assert(contains(V) && "Expected Value in map");
    getKnownBits(V).setAllOnes();
  }
  LLVM_ABI_FOR_TEST bool isAllConflict(const Value *V) const {
    return at(V).isAllConflict();
  }
  LLVM_ABI_FOR_TEST ArrayRef<const Value *> getRoots() const;
  LLVM_ABI_FOR_TEST SmallVector<const Value *> getLeaves() const;
  LLVM_ABI_FOR_TEST void intersectWith(const Value *V, KnownBits Known) {
    assert(contains(V) && "Expected Value in map");
    KnownBits &K = getKnownBits(V);
    K = K.intersectWith(Known);
  }

public:
  LLVM_ABI KnownBitsDataflow(Function &F);
  LLVM_ABI KnownBitsDataflow(const KnownBitsDataflow &) = delete;
  LLVM_ABI KnownBitsDataflow &operator=(const KnownBitsDataflow &) = delete;

  LLVM_ABI bool empty() const {
    return DenseMap<const Value *, KnownBits>::empty();
  }
  LLVM_ABI size_t size() const {
    return DenseMap<const Value *, KnownBits>::size();
  }
  LLVM_ABI bool contains(const Value *V) const {
    return DenseMap<const Value *, KnownBits>::contains(V);
  }
  LLVM_ABI KnownBits at(const Value *V) const {
    return DenseMap<const Value *, KnownBits>::at(V);
  }

  /// Invalidates KnownBits corresponding to \p V, and all dependent values in
  /// dataflow, and returns the invalidated leaves.
  LLVM_ABI SmallVector<const Value *> invalidate(const Value *V);

  LLVM_ABI void print(raw_ostream &OS) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class KnownBitsCache : protected KnownBitsDataflow {
  KnownBits computeKnownBitsForHorizontalOperation(
      const Operator *I, const APInt &DemandedElts, const SimplifyQuery &Q,
      const function_ref<KnownBits(const KnownBits &, const KnownBits &)>
          KnownBitsFunc);
  void computeKnownBitsFromLerpPattern(const Value *Op0, const Value *Op1,
                                       const APInt &DemandedElts,
                                       KnownBits &KnownOut,
                                       const SimplifyQuery &Q);
  void computeKnownBitsAddSub(bool Add, const Value *Op0, const Value *Op1,
                              bool NSW, bool NUW, const APInt &DemandedElts,
                              KnownBits &KnownOut, KnownBits &Known2,
                              const SimplifyQuery &Q);
  void computeKnownBitsMul(const Value *Op0, const Value *Op1, bool NSW,
                           bool NUW, const APInt &DemandedElts,
                           KnownBits &Known, KnownBits &Known2,
                           const SimplifyQuery &Q);
  void computeKnownBitsFromShiftOperator(
      const Operator *I, const APInt &DemandedElts, KnownBits &Known,
      KnownBits &Known2, const SimplifyQuery &Q,
      function_ref<KnownBits(const KnownBits &, const KnownBits &, bool)> KF);
  KnownBits getKnownBitsFromAndXorOr(const Operator *I,
                                     const APInt &DemandedElts,
                                     const KnownBits &KnownLHS,
                                     const KnownBits &KnownRHS,
                                     const SimplifyQuery &Q);
  void computeKnownBitsFromOperator(const Operator *I,
                                    const APInt &DemandedElts, KnownBits &Known,
                                    const SimplifyQuery &Q);
  KnownBits computeKnownBits(const Value *V, const SimplifyQuery &Q);
  KnownBits computeKnownBits(const Value *V, const APInt &DemandedElts,
                             const SimplifyQuery &Q);
  void computeKnownBits(const Value *V, const APInt &DemandedElts,
                        KnownBits &Known, const SimplifyQuery &Q);
  void computeKnownBits(const Value *V, KnownBits &Known,
                        const SimplifyQuery &Q);
  void compute(ArrayRef<const Value *> Leaves);

public:
  KnownBitsCache(Function &F);
  LLVM_ABI KnownBits getOrCompute(const Value *V);
};
} // end namespace llvm

#endif // LLVM_ANALYSIS_SEMILATTICE_H
