//===---- SimpleRemoteMemoryMapper.cpp - Remote memory mapper ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SimpleRemoteMemoryMapper.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm::orc {

SimpleRemoteMemoryMapper::SimpleRemoteMemoryMapper(ExecutionSession &ES,
                                                   SimpleMemoryMapBindings B)
    : ES(ES), B(std::move(B)) {}

void SimpleRemoteMemoryMapper::reserve(size_t NumBytes,
                                       OnReservedFunction OnReserved) {
  B.Reserve(
      [NumBytes, OnReserved = std::move(OnReserved)](
          Expected<ExecutorAddr> Result) mutable {
        if (!Result)
          return OnReserved(Result.takeError());
        OnReserved(ExecutorAddrRange(*Result, NumBytes));
      },
      ES, B.Instance, static_cast<uint64_t>(NumBytes));
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

  B.Initialize(std::move(OnInitialized), ES, B.Instance, std::move(FR));
}

void SimpleRemoteMemoryMapper::deinitialize(
    ArrayRef<ExecutorAddr> Allocations,
    MemoryMapper::OnDeinitializedFunction OnDeinitialized) {
  B.Deinitialize(std::move(OnDeinitialized), ES, B.Instance, Allocations);
}

void SimpleRemoteMemoryMapper::release(ArrayRef<ExecutorAddr> Bases,
                                       OnReleasedFunction OnReleased) {
  B.Release(std::move(OnReleased), ES, B.Instance, Bases);
}

} // namespace llvm::orc
