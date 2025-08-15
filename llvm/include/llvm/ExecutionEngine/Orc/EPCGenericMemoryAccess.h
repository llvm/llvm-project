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

class EPCGenericMemoryAccess : public MemoryAccess {
public:
  /// Function addresses for memory access.
  struct FuncAddrs {
    ExecutorAddr WriteUInt8s;
    ExecutorAddr WriteUInt16s;
    ExecutorAddr WriteUInt32s;
    ExecutorAddr WriteUInt64s;
    ExecutorAddr WritePointers;
    ExecutorAddr WriteBuffers;
    ExecutorAddr ReadUInt8s;
    ExecutorAddr ReadUInt16s;
    ExecutorAddr ReadUInt32s;
    ExecutorAddr ReadUInt64s;
    ExecutorAddr ReadPointers;
    ExecutorAddr ReadBuffers;
    ExecutorAddr ReadStrings;
  };

  /// Create an EPCGenericMemoryAccess instance from a given set of
  /// function addrs.
  EPCGenericMemoryAccess(ExecutorProcessControl &EPC, FuncAddrs FAs)
      : EPC(EPC), FAs(FAs) {}

  void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                        WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt8Write>)>(
        FAs.WriteUInt8s, std::move(OnWriteComplete), Ws);
  }

  void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt16Write>)>(
        FAs.WriteUInt16s, std::move(OnWriteComplete), Ws);
  }

  void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt32Write>)>(
        FAs.WriteUInt32s, std::move(OnWriteComplete), Ws);
  }

  void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessUInt64Write>)>(
        FAs.WriteUInt64s, std::move(OnWriteComplete), Ws);
  }

  void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                          WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessPointerWrite>)>(
        FAs.WritePointers, std::move(OnWriteComplete), Ws);
  }

  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnWriteComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<void(SPSSequence<SPSMemoryAccessBufferWrite>)>(
        FAs.WriteBuffers, std::move(OnWriteComplete), Ws);
  }

  void readUInt8sAsync(ArrayRef<ExecutorAddr> Rs,
                       OnReadUIntsCompleteFn<uint8_t> OnComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<SPSSequence<uint8_t>(SPSSequence<SPSExecutorAddr>)>(
        FAs.ReadUInt8s,
        [OnComplete = std::move(OnComplete)](
            Error Err, ReadUIntsResult<uint8_t> Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readUInt16sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint16_t> OnComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<SPSSequence<uint16_t>(
        SPSSequence<SPSExecutorAddr>)>(
        FAs.ReadUInt16s,
        [OnComplete = std::move(OnComplete)](
            Error Err, ReadUIntsResult<uint16_t> Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readUInt32sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint32_t> OnComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<SPSSequence<uint32_t>(
        SPSSequence<SPSExecutorAddr>)>(
        FAs.ReadUInt32s,
        [OnComplete = std::move(OnComplete)](
            Error Err, ReadUIntsResult<uint32_t> Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readUInt64sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint64_t> OnComplete) override {
    using namespace shared;
    EPC.callSPSWrapperAsync<SPSSequence<uint64_t>(
        SPSSequence<SPSExecutorAddr>)>(
        FAs.ReadUInt64s,
        [OnComplete = std::move(OnComplete)](
            Error Err, ReadUIntsResult<uint64_t> Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readPointersAsync(ArrayRef<ExecutorAddr> Rs,
                         OnReadPointersCompleteFn OnComplete) override {
    using namespace shared;
    using SPSSig = SPSSequence<SPSExecutorAddr>(SPSSequence<SPSExecutorAddr>);
    EPC.callSPSWrapperAsync<SPSSig>(
        FAs.ReadPointers,
        [OnComplete = std::move(OnComplete)](
            Error Err, ReadPointersResult Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readBuffersAsync(ArrayRef<ExecutorAddrRange> Rs,
                        OnReadBuffersCompleteFn OnComplete) override {
    using namespace shared;
    using SPSSig =
        SPSSequence<SPSSequence<uint8_t>>(SPSSequence<SPSExecutorAddrRange>);
    EPC.callSPSWrapperAsync<SPSSig>(
        FAs.ReadBuffers,
        [OnComplete = std::move(OnComplete)](Error Err,
                                             ReadBuffersResult Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

  void readStringsAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadStringsCompleteFn OnComplete) override {
    using namespace shared;
    using SPSSig = SPSSequence<SPSString>(SPSSequence<SPSExecutorAddr>);
    EPC.callSPSWrapperAsync<SPSSig>(
        FAs.ReadStrings,
        [OnComplete = std::move(OnComplete)](Error Err,
                                             ReadStringsResult Result) mutable {
          if (Err)
            OnComplete(std::move(Err));
          else
            OnComplete(std::move(Result));
        },
        Rs);
  }

private:
  ExecutorProcessControl &EPC;
  FuncAddrs FAs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
