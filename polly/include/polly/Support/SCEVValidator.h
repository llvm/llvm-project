//===--- SCEVValidator.h - Detect Scops -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Checks if a SCEV expression represents a valid affine expression.
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCEV_VALIDATOR_H
#define POLLY_SCEV_VALIDATOR_H

#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SetVector.h"

namespace llvm {
class Region;
class SCEV;
class SCEVConstant;
class ScalarEvolution;
class Value;
class Loop;
class LoadInst;
class CallInst;
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
/// @param S The SCEV to analyze.
/// @param R The region in which we look for dependences.
/// @param Scope Location where the value is needed.
/// @param AllowLoops Whether loop recurrences outside the loop that are in the
///                   region count as dependence.
bool hasScalarDepsInsideRegion(const llvm::SCEV *S, const llvm::Region *R,
                               llvm::Loop *Scope, bool AllowLoops);
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
} // namespace polly

#endif
