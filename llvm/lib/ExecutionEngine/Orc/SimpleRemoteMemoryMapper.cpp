//===---- SimpleRemoteMemoryMapper.cpp - Remote memory mapper ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SimpleRemoteMemoryMapper.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

namespace llvm::orc {

SimpleRemoteMemoryMapper::SimpleRemoteMemoryMapper(ExecutorProcessControl &EPC,
                                                   SymbolAddrs SAs)
    : EPC(EPC), SAs(SAs) {}

void SimpleRemoteMemoryMapper::reserve(size_t NumBytes,
                                       OnReservedFunction OnReserved) {
  EPC.callSPSWrapperAsync<rt::SPSSimpleRemoteMemoryMapReserveSignature>(
      SAs.Reserve,
      [NumBytes, OnReserved = std::move(OnReserved)](
          Error SerializationErr, Expected<ExecutorAddr> Result) mutable {
        if (SerializationErr) {
          cantFail(Result.takeError());
          return OnReserved(std::move(SerializationErr));
        }

        if (Result)
          OnReserved(ExecutorAddrRange(*Result, NumBytes));
        else
          OnReserved(Result.takeError());
      },
      SAs.Instance, static_cast<uint64_t>(NumBytes));
}

char *SimpleRemoteMemoryMapper::prepare(jitlink::LinkGraph &G,
                                        ExecutorAddr Addr, size_t ContentSize) {
  return G.allocateBuffer(ContentSize).data();
}

void SimpleRemoteMemoryMapper::initialize(MemoryMapper::AllocInfo &AI,
                                          OnInitializedFunction OnInitialized) {

  tpctypes::FinalizeRequest FR;

  std::swap(FR.Actions, AI.Actions);
  FR.Segments.reserve(AI.Segments.size());

  for (auto Seg : AI.Segments)
    FR.Segments.push_back({Seg.AG, AI.MappingBase + Seg.Offset,
                           Seg.ContentSize + Seg.ZeroFillSize,
                           ArrayRef<char>(Seg.WorkingMem, Seg.ContentSize)});

  EPC.callSPSWrapperAsync<rt::SPSSimpleRemoteMemoryMapInitializeSignature>(
      SAs.Initialize,
      [OnInitialized = std::move(OnInitialized)](
          Error SerializationErr, Expected<ExecutorAddr> Result) mutable {
        if (SerializationErr) {
          cantFail(Result.takeError());
          return OnInitialized(std::move(SerializationErr));
        }

        OnInitialized(std::move(Result));
      },
      SAs.Instance, std::move(FR));
}

void SimpleRemoteMemoryMapper::deinitialize(
    ArrayRef<ExecutorAddr> Allocations,
    MemoryMapper::OnDeinitializedFunction OnDeinitialized) {
  EPC.callSPSWrapperAsync<rt::SPSSimpleRemoteMemoryMapDeinitializeSignature>(
      SAs.Deinitialize,
      [OnDeinitialized = std::move(OnDeinitialized)](Error SerializationErr,
                                                     Error Result) mutable {
        if (SerializationErr) {
          cantFail(std::move(Result));
          return OnDeinitialized(std::move(SerializationErr));
        }

        OnDeinitialized(std::move(Result));
      },
      SAs.Instance, Allocations);
}

void SimpleRemoteMemoryMapper::release(ArrayRef<ExecutorAddr> Bases,
                                       OnReleasedFunction OnReleased) {
  EPC.callSPSWrapperAsync<rt::SPSSimpleRemoteMemoryMapReleaseSignature>(
      SAs.Release,
      [OnReleased = std::move(OnReleased)](Error SerializationErr,
                                           Error Result) mutable {
        if (SerializationErr) {
          cantFail(std::move(Result));
          return OnReleased(std::move(SerializationErr));
        }

        return OnReleased(std::move(Result));
      },
      SAs.Instance, Bases);
}

} // namespace llvm::orc
