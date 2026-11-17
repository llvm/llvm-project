//===- EPCGenericMemoryAccess.h - Generic EPC MemoryAccess impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the MemoryAccess interface by calling executor-side wrapper
// functions through Proxy objects.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
// This header is protocol-agnostic. To build an instance that targets the ORC
// runtime's SPS controller interface, see EPCGenericMemoryAccessSPS.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
namespace orc {

class EPCGenericMemoryAccess : public MemoryAccess {
public:
  using WriteUInt8sProxy = Proxy<void(ArrayRef<tpctypes::UInt8Write>)>;
  using WriteUInt16sProxy = Proxy<void(ArrayRef<tpctypes::UInt16Write>)>;
  using WriteUInt32sProxy = Proxy<void(ArrayRef<tpctypes::UInt32Write>)>;
  using WriteUInt64sProxy = Proxy<void(ArrayRef<tpctypes::UInt64Write>)>;
  using WritePointersProxy = Proxy<void(ArrayRef<tpctypes::PointerWrite>)>;
  using WriteBuffersProxy = Proxy<void(ArrayRef<tpctypes::BufferWrite>)>;
  using ReadUInt8sProxy = Proxy<std::vector<uint8_t>(ArrayRef<ExecutorAddr>)>;
  using ReadUInt16sProxy = Proxy<std::vector<uint16_t>(ArrayRef<ExecutorAddr>)>;
  using ReadUInt32sProxy = Proxy<std::vector<uint32_t>(ArrayRef<ExecutorAddr>)>;
  using ReadUInt64sProxy = Proxy<std::vector<uint64_t>(ArrayRef<ExecutorAddr>)>;
  using ReadPointersProxy =
      Proxy<std::vector<ExecutorAddr>(ArrayRef<ExecutorAddr>)>;
  using ReadBuffersProxy =
      Proxy<std::vector<std::vector<uint8_t>>(ArrayRef<ExecutorAddrRange>)>;
  using ReadStringsProxy =
      Proxy<std::vector<std::string>(ArrayRef<ExecutorAddr>)>;

  /// Proxies for the executor-side memory-access functions. These are
  /// protocol-agnostic: sps::createEPCGenericMemoryAccess populates them for
  /// the runtime's SPS controller interface, but a client targeting a different
  /// protocol can build its own Funcs and pass them to the constructor.
  struct Funcs {
    WriteUInt8sProxy WriteUInt8s;
    WriteUInt16sProxy WriteUInt16s;
    WriteUInt32sProxy WriteUInt32s;
    WriteUInt64sProxy WriteUInt64s;
    WritePointersProxy WritePointers;
    WriteBuffersProxy WriteBuffers;
    ReadUInt8sProxy ReadUInt8s;
    ReadUInt16sProxy ReadUInt16s;
    ReadUInt32sProxy ReadUInt32s;
    ReadUInt64sProxy ReadUInt64s;
    ReadPointersProxy ReadPointers;
    ReadBuffersProxy ReadBuffers;
    ReadStringsProxy ReadStrings;
  };

  /// Create an EPCGenericMemoryAccess instance from a given set of memory
  /// access proxies.
  EPCGenericMemoryAccess(ExecutionSession &ES, Funcs Fns)
      : ES(ES), Fns(std::move(Fns)) {}

  void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                        WriteResultFn OnWriteComplete) override {
    Fns.WriteUInt8s(std::move(OnWriteComplete), ES, Ws);
  }

  void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    Fns.WriteUInt16s(std::move(OnWriteComplete), ES, Ws);
  }

  void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    Fns.WriteUInt32s(std::move(OnWriteComplete), ES, Ws);
  }

  void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                         WriteResultFn OnWriteComplete) override {
    Fns.WriteUInt64s(std::move(OnWriteComplete), ES, Ws);
  }

  void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                          WriteResultFn OnWriteComplete) override {
    Fns.WritePointers(std::move(OnWriteComplete), ES, Ws);
  }

  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnWriteComplete) override {
    Fns.WriteBuffers(std::move(OnWriteComplete), ES, Ws);
  }

  void readUInt8sAsync(ArrayRef<ExecutorAddr> Rs,
                       OnReadUIntsCompleteFn<uint8_t> OnComplete) override {
    Fns.ReadUInt8s(std::move(OnComplete), ES, Rs);
  }

  void readUInt16sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint16_t> OnComplete) override {
    Fns.ReadUInt16s(std::move(OnComplete), ES, Rs);
  }

  void readUInt32sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint32_t> OnComplete) override {
    Fns.ReadUInt32s(std::move(OnComplete), ES, Rs);
  }

  void readUInt64sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint64_t> OnComplete) override {
    Fns.ReadUInt64s(std::move(OnComplete), ES, Rs);
  }

  void readPointersAsync(ArrayRef<ExecutorAddr> Rs,
                         OnReadPointersCompleteFn OnComplete) override {
    Fns.ReadPointers(std::move(OnComplete), ES, Rs);
  }

  void readBuffersAsync(ArrayRef<ExecutorAddrRange> Rs,
                        OnReadBuffersCompleteFn OnComplete) override {
    Fns.ReadBuffers(std::move(OnComplete), ES, Rs);
  }

  void readStringsAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadStringsCompleteFn OnComplete) override {
    Fns.ReadStrings(std::move(OnComplete), ES, Rs);
  }

private:
  ExecutionSession &ES;
  Funcs Fns;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
