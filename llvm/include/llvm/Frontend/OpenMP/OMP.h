//===-- OMP.h - Core OpenMP definitions and declarations ---------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the core set of OpenMP definitions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMP_H
#define LLVM_FRONTEND_OPENMP_OMP_H

#include "llvm/Frontend/OpenMP/OMP.h.inc"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm::omp {
ArrayRef<Directive> getLeafConstructs(Directive D);
ArrayRef<Directive> getLeafConstructsOrSelf(Directive D);

ArrayRef<Directive>
getLeafOrCompositeConstructs(Directive D, SmallVectorImpl<Directive> &Output);

Directive getCompoundConstruct(ArrayRef<Directive> Parts);

bool isLeafConstruct(Directive D);
bool isCompositeConstruct(Directive D);
bool isCombinedConstruct(Directive D);

/// Can clause C have an iterator-modifier.
static constexpr inline bool canHaveIterator(Clause C) {
  // [5.2:67:5]
  switch (C) {
  case OMPC_affinity:
  case OMPC_depend:
  case OMPC_from:
  case OMPC_map:
  case OMPC_to:
    return true;
  default:
    return false;
  }
}

ArrayRef<unsigned> getOpenMPVersions();

/// Create a nicer version of a function name for humans to look at.
std::string prettifyFunctionName(StringRef FunctionName);

/// Deconstruct an OpenMP kernel name into the parent function name and the line
/// number.
std::string deconstructOpenMPKernelName(StringRef KernelName, unsigned &LineNo);

} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMP_H
