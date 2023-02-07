//===-- Vectorize.h - Vectorization Transformations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Vectorize transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_H

namespace llvm {
class BasicBlock;
class Pass;

//===----------------------------------------------------------------------===//
/// Vectorize configuration.
struct VectorizeConfig {
  //===--------------------------------------------------------------------===//
  // Target architecture related parameters

  /// The size of the native vector registers.
  unsigned VectorBits;

  /// Vectorize boolean values.
  bool VectorizeBools;

  /// Vectorize integer values.
  bool VectorizeInts;

  /// Vectorize floating-point values.
  bool VectorizeFloats;

  /// Vectorize pointer values.
  bool VectorizePointers;

  /// Vectorize casting (conversion) operations.
  bool VectorizeCasts;

  /// Vectorize floating-point math intrinsics.
  bool VectorizeMath;

  /// Vectorize bit intrinsics.
  bool VectorizeBitManipulations;

  /// Vectorize the fused-multiply-add intrinsic.
  bool VectorizeFMA;

  /// Vectorize select instructions.
  bool VectorizeSelect;

  /// Vectorize comparison instructions.
  bool VectorizeCmp;

  /// Vectorize getelementptr instructions.
  bool VectorizeGEP;

  /// Vectorize loads and stores.
  bool VectorizeMemOps;

  /// Only generate aligned loads and stores.
  bool AlignedOnly;

  //===--------------------------------------------------------------------===//
  // Misc parameters

  /// The required chain depth for vectorization.
  unsigned ReqChainDepth;

  /// The maximum search distance for instruction pairs.
  unsigned SearchLimit;

  /// The maximum number of candidate pairs with which to use a full
  ///        cycle check.
  unsigned MaxCandPairsForCycleCheck;

  /// Replicating one element to a pair breaks the chain.
  bool SplatBreaksChain;

  /// The maximum number of pairable instructions per group.
  unsigned MaxInsts;

  /// The maximum number of candidate instruction pairs per group.
  unsigned MaxPairs;

  /// The maximum number of pairing iterations.
  unsigned MaxIter;

  /// Don't try to form odd-length vectors.
  bool Pow2LenOnly;

  /// Don't boost the chain-depth contribution of loads and stores.
  bool NoMemOpBoost;

  /// Use a fast instruction dependency analysis.
  bool FastDep;

  /// Initialize the VectorizeConfig from command line options.
  VectorizeConfig();
};

//===----------------------------------------------------------------------===//
//
// LoopVectorize - Create a loop vectorization pass.
//
Pass *createLoopVectorizePass();
Pass *createLoopVectorizePass(bool InterleaveOnlyWhenForced,
                              bool VectorizeOnlyWhenForced);

//===----------------------------------------------------------------------===//
//
// SLPVectorizer - Create a bottom-up SLP vectorizer pass.
//
Pass *createSLPVectorizerPass();

//===----------------------------------------------------------------------===//
/// Vectorize the BasicBlock.
///
/// @param BB The BasicBlock to be vectorized
/// @param P  The current running pass, should require AliasAnalysis and
///           ScalarEvolution. After the vectorization, AliasAnalysis,
///           ScalarEvolution and CFG are preserved.
///
/// @return True if the BB is changed, false otherwise.
///
bool vectorizeBasicBlock(Pass *P, BasicBlock &BB,
                         const VectorizeConfig &C = VectorizeConfig());

//===----------------------------------------------------------------------===//
//
// LoadStoreVectorizer - Create vector loads and stores, but leave scalar
// operations.
//
Pass *createLoadStoreVectorizerPass();

//===----------------------------------------------------------------------===//
//
// Optimize partial vector operations using target cost models.
//
Pass *createVectorCombinePass();

} // End llvm namespace

#endif
