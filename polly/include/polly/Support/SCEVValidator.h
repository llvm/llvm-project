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
#include <vector>

namespace llvm {
class Region;
class SCEV;
class ScalarEvolution;
class Value;
class Loop;
class LoadInst;
}

namespace polly {
/// @brief Find the loops referenced from a SCEV expression.
///
/// @param Expr The SCEV expression to scan for loops.
/// @param Loops A vector into which the found loops are inserted.
void findLoops(const llvm::SCEV *Expr,
               llvm::SetVector<const llvm::Loop *> &Loops);

/// @brief Find the values referenced by SCEVUnknowns in a given SCEV
/// expression.
///
/// @param Expr The SCEV expression to scan for SCEVUnknowns.
/// @param Expr A vector into which the found values are inserted.
void findValues(const llvm::SCEV *Expr, llvm::SetVector<llvm::Value *> &Values);

/// Returns true when the SCEV contains references to instructions within the
/// region.
///
/// @param S The SCEV to analyze.
/// @param R The region in which we look for dependences.
bool hasScalarDepsInsideRegion(const llvm::SCEV *S, const llvm::Region *R);
bool isAffineExpr(const llvm::Region *R, const llvm::SCEV *Expression,
                  llvm::ScalarEvolution &SE, const llvm::Value *BaseAddress = 0,
                  InvariantLoadsSetTy *ILS = nullptr);

/// @brief Check if @p V describes an affine parameter constraint in @p R.
bool isAffineParamConstraint(llvm::Value *V, const llvm::Region *R,
                             llvm::ScalarEvolution &SE,
                             std::vector<const llvm::SCEV *> &Params,
                             bool OrExpr = false);

std::vector<const llvm::SCEV *>
getParamsInAffineExpr(const llvm::Region *R, const llvm::SCEV *Expression,
                      llvm::ScalarEvolution &SE,
                      const llvm::Value *BaseAddress = 0);

/// @brief Extract the constant factors from the multiplication @p M.
///
/// @param M  A potential SCEV multiplication.
/// @param SE The ScalarEvolution analysis to create new SCEVs.
///
/// @returns The constant factor in @p M and the rest of @p M.
std::pair<const llvm::SCEV *, const llvm::SCEV *>
extractConstantFactor(const llvm::SCEV *M, llvm::ScalarEvolution &SE);
}

#endif
