//===------------------- CommonOrcRuntimeTypes.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic types usable with SPS and the ORC runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_COMMONORCRUNTIMETYPES_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_COMMONORCRUNTIMETYPES_H

#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"

namespace llvm {
namespace orc {
namespace shared {

/// Represents an address range in the exceutor process.
struct ExecutorAddressRange {
  ExecutorAddressRange() = default;
  ExecutorAddressRange(JITTargetAddress StartAddress,
                       JITTargetAddress EndAddress)
      : StartAddress(StartAddress), EndAddress(EndAddress) {}

  bool empty() const { return StartAddress == EndAddress; }
  size_t size() const { return EndAddress - StartAddress; }

  JITTargetAddress StartAddress = 0;
  JITTargetAddress EndAddress = 0;
};

using SPSExecutorAddressRange =
    SPSTuple<SPSExecutorAddress, SPSExecutorAddress>;

/// Serialization traits for address ranges.
template <>
class SPSSerializationTraits<SPSExecutorAddressRange, ExecutorAddressRange> {
public:
  static size_t size(const ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::size(
        Value.StartAddress, Value.EndAddress);
  }

  static bool serialize(SPSOutputBuffer &BOB,
                        const ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::serialize(
        BOB, Value.StartAddress, Value.EndAddress);
  }

  static bool deserialize(SPSInputBuffer &BIB, ExecutorAddressRange &Value) {
    return SPSArgList<SPSExecutorAddress, SPSExecutorAddress>::deserialize(
        BIB, Value.StartAddress, Value.EndAddress);
  }
};

using SPSExecutorAddressRangeSequence = SPSSequence<SPSExecutorAddressRange>;

} // End namespace shared.
} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_COMMONORCRUNTIMETYPES_H
