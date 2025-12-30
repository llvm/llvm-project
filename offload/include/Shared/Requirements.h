//===-- OpenMP/Requirements.h - User required requirements -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handling of the `omp requires` directive, e.g., requiring unified shared
// memory.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_REQUIREMENTS_H
#define OMPTARGET_OPENMP_REQUIREMENTS_H

#include "Shared/Debug.h"

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cstdint>

enum OpenMPOffloadingRequiresDirFlags : int64_t {
  /// flag undefined.
  OMP_REQ_UNDEFINED = 0x000,
  /// no requires directive present.
  OMP_REQ_NONE = 0x001,
  /// reverse_offload clause.
  OMP_REQ_REVERSE_OFFLOAD = 0x002,
  /// unified_address clause.
  OMP_REQ_UNIFIED_ADDRESS = 0x004,
  /// unified_shared_memory clause.
  OMP_REQ_UNIFIED_SHARED_MEMORY = 0x008,
  /// dynamic_allocators clause.
  OMP_REQ_DYNAMIC_ALLOCATORS = 0x010,
  /// Auto zero-copy extension:
  /// when running on an APU, the GPU plugin may decide to
  /// run in zero-copy even though the user did not program
  /// their application with unified_shared_memory requirement.
  OMPX_REQ_AUTO_ZERO_COPY = 0x020
};

class RequirementCollection {
  int64_t SetFlags = OMP_REQ_UNDEFINED;

  /// Check consistency between different requires flags (from different
  /// translation units).
  void checkConsistency(int64_t NewFlags, int64_t SetFlags,
                        OpenMPOffloadingRequiresDirFlags Flag,
                        llvm::StringRef Clause) {
    if ((SetFlags & Flag) != (NewFlags & Flag)) {
      FATAL_MESSAGE(2, "'#pragma omp requires %s' not used consistently!",
                    Clause.data());
    }
  }

public:
  /// Register \p NewFlags as part of the user requirements.
  void addRequirements(int64_t NewFlags) {
    // TODO: add more elaborate check.
    // Minimal check: only set requires flags if previous value
    // is undefined. This ensures that only the first call to this
    // function will set the requires flags. All subsequent calls
    // will be checked for compatibility.
    assert(NewFlags != OMP_REQ_UNDEFINED &&
           "illegal undefined flag for requires directive!");
    if (SetFlags == OMP_REQ_UNDEFINED) {
      SetFlags = NewFlags;
      return;
    }

    // Auto zero-copy is only valid when no other requirement has been set
    // and it is computed at device initialization time, after the requirement
    // flag has already been set to OMP_REQ_NONE.
    if (SetFlags == OMP_REQ_NONE && NewFlags == OMPX_REQ_AUTO_ZERO_COPY) {
      SetFlags = NewFlags;
      return;
    }

    // If multiple compilation units are present enforce
    // consistency across all of them for require clauses:
    //  - reverse_offload
    //  - unified_address
    //  - unified_shared_memory
    //  - dynamic_allocators
    checkConsistency(NewFlags, SetFlags, OMP_REQ_REVERSE_OFFLOAD,
                     "reverse_offload");
    checkConsistency(NewFlags, SetFlags, OMP_REQ_UNIFIED_ADDRESS,
                     "unified_address");
    checkConsistency(NewFlags, SetFlags, OMP_REQ_UNIFIED_SHARED_MEMORY,
                     "unified_shared_memory");
    checkConsistency(NewFlags, SetFlags, OMP_REQ_DYNAMIC_ALLOCATORS,
                     "dynamic_allocators");
  }

  /// Return the user provided requirements.
  int64_t getRequirements() const { return SetFlags; }
};

#endif // OMPTARGET_OPENMP_DEVICE_REQUIREMENTS_H
