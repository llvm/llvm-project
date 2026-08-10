//===- EPCGenericMemoryAccess.h - Generic EPC MemoryAccess impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the MemoryAccess interface by calling executor-side wrapper
// functions through rt::Proxy objects.
//
// This simplifies the implementaton of new ExecutorProcessControl instances,
// as this implementation will always work (at the cost of some performance
// overhead for the calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICMEMORYACCESS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpecs.h"

namespace llvm {
namespace orc {

class EPCGenericMemoryAccess : public MemoryAccess {
public:
  /// Proxies for the executor-side memory-access functions. These are
  /// protocol-agnostic: EPCGenericMemoryAccess::Create populates them for the
  /// runtime's SPS controller interface, but a client targeting a different
  /// protocol can build its own Funcs and pass them to the constructor.
  struct Funcs {
    rt::MemWriteUInt8sProxy WriteUInt8s;
    rt::MemWriteUInt16sProxy WriteUInt16s;
    rt::MemWriteUInt32sProxy WriteUInt32s;
    rt::MemWriteUInt64sProxy WriteUInt64s;
    rt::MemWritePointersProxy WritePointers;
    rt::MemWriteBuffersProxy WriteBuffers;
    rt::MemReadUInt8sProxy ReadUInt8s;
    rt::MemReadUInt16sProxy ReadUInt16s;
    rt::MemReadUInt32sProxy ReadUInt32s;
    rt::MemReadUInt64sProxy ReadUInt64s;
    rt::MemReadPointersProxy ReadPointers;
    rt::MemReadBuffersProxy ReadBuffers;
    rt::MemReadStringsProxy ReadStrings;
  };

  /// Create an EPCGenericMemoryAccess instance that reaches the memory-access
  /// wrappers in ES's bootstrap JITDylib via the runtime's SPS controller
  /// interface.
  static Expected<std::unique_ptr<MemoryAccess>> Create(ExecutionSession &ES) {
    namespace sps = rt::sps;
    Funcs Fns;
    if (auto Err = rt::buildProxies(
            ES, rt::proxyInit<sps::MemWriteUInt8sProxySpec>(&Fns.WriteUInt8s),
            rt::proxyInit<sps::MemWriteUInt16sProxySpec>(&Fns.WriteUInt16s),
            rt::proxyInit<sps::MemWriteUInt32sProxySpec>(&Fns.WriteUInt32s),
            rt::proxyInit<sps::MemWriteUInt64sProxySpec>(&Fns.WriteUInt64s),
            rt::proxyInit<sps::MemWritePointersProxySpec>(&Fns.WritePointers),
            rt::proxyInit<sps::MemWriteBuffersProxySpec>(&Fns.WriteBuffers),
            rt::proxyInit<sps::MemReadUInt8sProxySpec>(&Fns.ReadUInt8s),
            rt::proxyInit<sps::MemReadUInt16sProxySpec>(&Fns.ReadUInt16s),
            rt::proxyInit<sps::MemReadUInt32sProxySpec>(&Fns.ReadUInt32s),
            rt::proxyInit<sps::MemReadUInt64sProxySpec>(&Fns.ReadUInt64s),
            rt::proxyInit<sps::MemReadPointersProxySpec>(&Fns.ReadPointers),
            rt::proxyInit<sps::MemReadBuffersProxySpec>(&Fns.ReadBuffers),
            rt::proxyInit<sps::MemReadStringsProxySpec>(&Fns.ReadStrings)))
      return std::move(Err);
    return std::make_unique<EPCGenericMemoryAccess>(ES, std::move(Fns));
  }

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
