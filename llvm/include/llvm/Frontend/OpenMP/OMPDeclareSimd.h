//===- OMPDeclareSimd.h - OpenMP declare simd related types and helpers ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines types and helpers used when dealing with OpenMP declare simd.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPDECLARESIMD_H
#define LLVM_FRONTEND_OPENMP_OMPDECLARESIMD_H

#include "llvm/ADT/APSInt.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace omp {

/// Kind of parameter in a function with 'declare simd' directive.
enum class DeclareSimdKindTy {
  Linear,
  LinearRef,
  LinearUVal,
  LinearVal,
  Uniform,
  Vector,
};

/// Attribute set of the `declare simd` parameter.
struct DeclareSimdAttrTy {
  DeclareSimdKindTy Kind = DeclareSimdKindTy::Vector;
  llvm::APSInt StrideOrArg;
  llvm::APSInt Alignment;
  bool HasVarStride = false;
};

/// Type of branch clause of the `declare simd` directive.
enum class DeclareSimdBranch {
  Undefined,
  Notinbranch,
  Inbranch,
};

std::string mangleVectorParameters(llvm::ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs);

void emitDeclareSimdFunction(llvm::Function *Fn,
                             const llvm::APSInt &VLENVal,
                             llvm::ArrayRef<llvm::omp::DeclareSimdAttrTy> ParamAttrs,
                             DeclareSimdBranch Branch);

} // end namespace omp

} // end namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPDECLARESIMD_H
