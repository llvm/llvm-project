//===- AllocationActions.h -- JITLink allocation support calls  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for making memory allocation support calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_ALLOCATIONACTIONS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_ALLOCATIONACTIONS_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/Memory.h"

#include <vector>

namespace llvm {
namespace orc {
namespace shared {

/// A pair of WrapperFunctionCalls, one to be run at finalization time, one to
/// be run at deallocation time.
///
/// AllocActionCallPairs should be constructed for paired operations (e.g.
/// __register_ehframe and __deregister_ehframe for eh-frame registration).
/// See comments for AllocActions for execution ordering.
///
/// For unpaired operations one or the other member can be left unused, as
/// AllocationActionCalls with an FnAddr of zero will be skipped.
struct AllocActionCallPair {
  WrapperFunctionCall Finalize;
  WrapperFunctionCall Dealloc;
};

/// A vector of allocation actions to be run for this allocation.
///
/// Finalize allocations will be run in order at finalize time. Dealloc
/// actions will be run in reverse order at deallocation time.
using AllocActions = std::vector<AllocActionCallPair>;

using SPSAllocActionCallPair =
    SPSTuple<SPSWrapperFunctionCall, SPSWrapperFunctionCall>;

template <>
class SPSSerializationTraits<SPSAllocActionCallPair,
                             AllocActionCallPair> {
  using AL = SPSAllocActionCallPair::AsArgList;

public:
  static size_t size(const AllocActionCallPair &AAP) {
    return AL::size(AAP.Finalize, AAP.Dealloc);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const AllocActionCallPair &AAP) {
    return AL::serialize(OB, AAP.Finalize, AAP.Dealloc);
  }

  static bool deserialize(SPSInputBuffer &IB,
                          AllocActionCallPair &AAP) {
    return AL::deserialize(IB, AAP.Finalize, AAP.Dealloc);
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_ALLOCATIONACTIONS_H
