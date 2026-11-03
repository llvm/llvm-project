//===- OmptEventInfoTy.h - OMPT specific trace record data ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data structure used to communicate OMPT specific profiler data from the
// high-level libomptarget into the vendor-specific plugins
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_INCLUDE_OPENMP_OMPT_OMPTEVENTINFOTY_H
#define OFFLOAD_INCLUDE_OPENMP_OMPT_OMPTEVENTINFOTY_H

#include "Shared/Debug.h"

struct ompt_record_ompt_t;

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// Holds info needed to fill asynchronous trace records
struct OmptEventInfoTy {
  /// The granted number of teams at runtime
  uint64_t NumTeams;
  /// Pointer to the actual buffer storage location
  ompt_record_ompt_t *TraceRecord;
};

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_INCLUDE_OPENMP_OMPT_OMPTEVENTINFOTY_H
