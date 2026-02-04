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
#include "llvm/Support/Compiler.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm::omp {
LLVM_ABI ArrayRef<Directive> getLeafConstructs(Directive D);
LLVM_ABI ArrayRef<Directive> getLeafConstructsOrSelf(Directive D);

LLVM_ABI ArrayRef<Directive>
getLeafOrCompositeConstructs(Directive D, SmallVectorImpl<Directive> &Output);

LLVM_ABI Directive getCompoundConstruct(ArrayRef<Directive> Parts);

LLVM_ABI bool isLeafConstruct(Directive D);
LLVM_ABI bool isCompositeConstruct(Directive D);
LLVM_ABI bool isCombinedConstruct(Directive D);

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

// Can clause C create a private copy of a variable.
static constexpr inline bool isPrivatizingClause(Clause C) {
  switch (C) {
  case OMPC_detach:
  case OMPC_firstprivate:
  // TODO case OMPC_induction:
  case OMPC_in_reduction:
  case OMPC_is_device_ptr:
  case OMPC_lastprivate:
  case OMPC_linear:
  case OMPC_private:
  case OMPC_reduction:
  case OMPC_task_reduction:
  case OMPC_use_device_ptr:
    return true;
  default:
    return false;
  }
}

static constexpr unsigned FallbackVersion = 52;
LLVM_ABI ArrayRef<unsigned> getOpenMPVersions();

/// Can directive D, under some circumstances, create a private copy
/// of a variable in given OpenMP version?
bool isPrivatizingConstruct(Directive D, unsigned Version);

/// Create a nicer version of a function name for humans to look at.
LLVM_ABI std::string prettifyFunctionName(StringRef FunctionName);

/// Deconstruct an OpenMP kernel name into the parent function name and the line
/// number.
LLVM_ABI std::string deconstructOpenMPKernelName(StringRef KernelName,
                                                 unsigned &LineNo);

} // namespace llvm::omp

#endif // LLVM_FRONTEND_OPENMP_OMP_H
