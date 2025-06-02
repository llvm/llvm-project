//===- HashRecognize.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface for the HashRecognize analysis, which identifies hash functions
// that can be optimized using a lookup-table or with target-specific
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_HASHRECOGNIZE_H
#define LLVM_ANALYSIS_HASHRECOGNIZE_H

#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/KnownBits.h"
#include <variant>

namespace llvm {

class LPMUpdater;

/// A tuple of bits that are expected to be zero, number N of them expected to
/// be zero, with a boolean indicating whether it's the top or bottom N bits
/// expected to be zero.
using ErrBits = std::tuple<KnownBits, unsigned, bool>;

/// A custom std::array with 256 entries, that also has a print function.
struct CRCTable : public std::array<APInt, 256> {
  void print(raw_ostream &OS) const;
};

/// The structure that is returned when a polynomial algorithm was recognized by
/// the analysis. Currently, only the CRC algorithm is recognized.
struct PolynomialInfo {
  // The small constant trip-count of the analyzed loop.
  unsigned TripCount;

  // The LHS in a polynomial operation, or the initial variable of the
  // computation, since all polynomial operations must have a constant RHS,
  // which is the generating polynomial. It is the LHS of the polynomial
  // division in the case of CRC. Since polynomial division is an XOR in
  // GF(2^m), this variable must be XOR'ed with RHS in a loop to yield the
  // ComputedValue.
  const Value *LHS;

  // The generating polynomial, or the RHS of the polynomial division in the
  // case of CRC.
  APInt RHS;

  // The final computed value. This is a remainder of a polynomial division in
  // the case of CRC, which must be zero.
  const Value *ComputedValue;

  // Set to true in the case of big-endian.
  bool ByteOrderSwapped;

  // An optional auxiliary checksum that augments the LHS. In the case of CRC,
  // it is XOR'ed with the LHS, so that the computation's final remainder is
  // zero.
  const Value *LHSAux;

  PolynomialInfo(unsigned TripCount, const Value *LHS, const APInt &RHS,
                 const Value *ComputedValue, bool ByteOrderSwapped,
                 const Value *LHSAux = nullptr);
};

/// The analysis.
class HashRecognize {
  const Loop &L;
  ScalarEvolution &SE;

public:
  HashRecognize(const Loop &L, ScalarEvolution &SE);

  // The main analysis entry point.
  std::variant<PolynomialInfo, ErrBits, StringRef> recognizeCRC() const;

  // Auxilary entry point after analysis to interleave the generating polynomial
  // and return a 256-entry CRC table.
  CRCTable genSarwateTable(const APInt &GenPoly, bool ByteOrderSwapped) const;

  void print(raw_ostream &OS) const;
};

class HashRecognizePrinterPass
    : public PassInfoMixin<HashRecognizePrinterPass> {
  raw_ostream &OS;

public:
  explicit HashRecognizePrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &);
};

class HashRecognizeAnalysis : public AnalysisInfoMixin<HashRecognizeAnalysis> {
  friend AnalysisInfoMixin<HashRecognizeAnalysis>;
  static AnalysisKey Key;

public:
  using Result = HashRecognize;
  Result run(Loop &L, LoopAnalysisManager &AM, LoopStandardAnalysisResults &AR);
};
} // namespace llvm

#endif
