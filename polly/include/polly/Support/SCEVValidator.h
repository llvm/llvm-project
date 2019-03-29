//===--- SCEVValidator.h - Detect Scops -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks if a SCEV expression represents a valid affine expression.
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCEV_VALIDATOR_H
#define POLLY_SCEV_VALIDATOR_H

#include "polly/Support/ScopHelper.h"

namespace llvm {
class SCEVConstant;
} // namespace llvm

namespace polly {

/// Check if a call is side-effect free and has only constant arguments.
///
/// Such calls can be re-generated easily, so we do not need to model them
/// as scalar dependences.
///
/// @param Call The call to check.
bool isConstCall(llvm::CallInst *Call);

/// Check if some parameters in the affine expression might hide induction
/// variables. If this is the case, we will try to delinearize the accesses
/// taking into account this information to possibly obtain a memory access
/// with more structure. Currently we assume that each parameter that
/// comes from a function call might depend on a (virtual) induction variable.
/// This covers calls to 'get_global_id' and 'get_local_id' as they commonly
/// arise in OpenCL code, while not catching any false-positives in our current
/// tests.
bool hasIVParams(const llvm::SCEV *Expr);

/// Find the loops referenced from a SCEV expression.
///
/// @param Expr The SCEV expression to scan for loops.
/// @param Loops A vector into which the found loops are inserted.
void findLoops(const llvm::SCEV *Expr,
               llvm::SetVector<const llvm::Loop *> &Loops);

/// Find the values referenced by SCEVUnknowns in a given SCEV
/// expression.
///
/// @param Expr   The SCEV expression to scan for SCEVUnknowns.
/// @param SE     The ScalarEvolution analysis for this function.
/// @param Values A vector into which the found values are inserted.
void findValues(const llvm::SCEV *Expr, llvm::ScalarEvolution &SE,
                llvm::SetVector<llvm::Value *> &Values);

/// Returns true when the SCEV contains references to instructions within the
/// region.
///
/// @param Expr The SCEV to analyze.
/// @param R The region in which we look for dependences.
/// @param Scope Location where the value is needed.
/// @param AllowLoops Whether loop recurrences outside the loop that are in the
///                   region count as dependence.
bool hasScalarDepsInsideRegion(const llvm::SCEV *Expr, const llvm::Region *R,
                               llvm::Loop *Scope, bool AllowLoops,
                               const InvariantLoadsSetTy &ILS);
bool isAffineExpr(const llvm::Region *R, llvm::Loop *Scope,
                  const llvm::SCEV *Expression, llvm::ScalarEvolution &SE,
                  InvariantLoadsSetTy *ILS = nullptr);

/// Check if @p V describes an affine constraint in @p R.
bool isAffineConstraint(llvm::Value *V, const llvm::Region *R,
                        llvm::Loop *Scope, llvm::ScalarEvolution &SE,
                        ParameterSetTy &Params, bool OrExpr = false);

ParameterSetTy getParamsInAffineExpr(const llvm::Region *R, llvm::Loop *Scope,
                                     const llvm::SCEV *Expression,
                                     llvm::ScalarEvolution &SE);

/// Extract the constant factors from the multiplication @p M.
///
/// @param M  A potential SCEV multiplication.
/// @param SE The ScalarEvolution analysis to create new SCEVs.
///
/// @returns The constant factor in @p M and the rest of @p M.
std::pair<const llvm::SCEVConstant *, const llvm::SCEV *>
extractConstantFactor(const llvm::SCEV *M, llvm::ScalarEvolution &SE);

/// Try to look through PHI nodes, where some incoming edges come from error
/// blocks.
///
/// In case a PHI node follows an error block we can assume that the incoming
/// value can only come from the node that is not an error block. As a result,
/// conditions that seemed non-affine before are now in fact affine.
const llvm::SCEV *tryForwardThroughPHI(const llvm::SCEV *Expr, llvm::Region &R,
                                       llvm::ScalarEvolution &SE,
                                       llvm::LoopInfo &LI,
                                       const llvm::DominatorTree &DT);

/// Return a unique non-error block incoming value for @p PHI if available.
///
/// @param R The region to run our code on.
/// @param LI The loopinfo tree
/// @param DT The dominator tree
llvm::Value *getUniqueNonErrorValue(llvm::PHINode *PHI, llvm::Region *R,
                                    llvm::LoopInfo &LI,
                                    const llvm::DominatorTree &DT);
} // namespace polly

#endif
