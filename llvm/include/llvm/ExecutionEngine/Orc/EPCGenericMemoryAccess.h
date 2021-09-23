//===- EPCGenericMemoryAccess.h - Generic EPC MemoryAccess impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements ExecutorProcessControl::MemoryAccess by making calls to
// ExecutorProcessControl::callWrapperAsync.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

class EPCGenericMemoryAccess : public ExecutorProcessControl::MemoryAccess {
public:
  /// Function addresses for memory access.
  struct FuncAddrs {
    ExecutorAddr WriteUInt8s;
    ExecutorAddr WriteUInt16s;
    ExecutorAddr WriteUInt32s;
    ExecutorAddr WriteUInt64s;
    ExecutorAddr WriteBuffers;
  };

  /// Create an EPCGenericMemoryAccess instance from a given set of
  /// function addrs.
  EPCGenericMemoryAccess(ExecutorProcessControl &EPC, FuncAddrs FAs)
      : EPC(EPC), FAs(FAs) {}

  void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                        WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt8Write>)>(
        std::move(OnWriteComplete), FAs.WriteUInt8s.getValue(), Ws);
  }

  void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt16Write>)>(
        std::move(OnWriteComplete), FAs.WriteUInt16s.getValue(), Ws);
  }

  void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt32Write>)>(
        std::move(OnWriteComplete), FAs.WriteUInt32s.getValue(), Ws);
  }

  void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt64Write>)>(
        std::move(OnWriteComplete), FAs.WriteUInt64s.getValue(), Ws);
  }

  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessBufferWrite>)>(
        std::move(OnWriteComplete), FAs.WriteBuffers.getValue(), Ws);
  }

private:
  ExecutorProcessControl &EPC;
  FuncAddrs FAs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
