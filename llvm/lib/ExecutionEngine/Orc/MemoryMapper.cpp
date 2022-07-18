//===- MemoryMapper.cpp - Cross-process memory mapper ------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MemoryMapper.h"

namespace llvm {
namespace orc {

MemoryMapper::~MemoryMapper() {}

void InProcessMemoryMapper::reserve(size_t NumBytes,
                                    OnReservedFunction OnReserved) {
  std::error_code EC;
  auto MB = sys::Memory::allocateMappedMemory(
      NumBytes, nullptr, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);

  if (EC)
    return OnReserved(errorCodeToError(EC));

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Reservations[MB.base()].Size = MB.allocatedSize();
  }

  OnReserved(
      ExecutorAddrRange(ExecutorAddr::fromPtr(MB.base()), MB.allocatedSize()));
}

char *InProcessMemoryMapper::prepare(ExecutorAddr Addr, size_t ContentSize) {
  return Addr.toPtr<char *>();
}

void InProcessMemoryMapper::initialize(MemoryMapper::AllocInfo &AI,
                                       OnInitializedFunction OnInitialized) {
  ExecutorAddr MinAddr(~0ULL);

  for (auto &Segment : AI.Segments) {
    auto Base = AI.MappingBase + Segment.Offset;
    auto Size = Segment.ContentSize + Segment.ZeroFillSize;

    if (Base < MinAddr)
      MinAddr = Base;

    std::memset((Base + Segment.ContentSize).toPtr<void *>(), 0,
                Segment.ZeroFillSize);

    if (auto EC = sys::Memory::protectMappedMemory({Base.toPtr<void *>(), Size},
                                                   Segment.Prot)) {
      return OnInitialized(errorCodeToError(EC));
    }
    if (Segment.Prot & sys::Memory::MF_EXEC)
      sys::Memory::InvalidateInstructionCache(Base.toPtr<void *>(), Size);
  }

  auto DeinitializeActions = shared::runFinalizeActions(AI.Actions);
  if (!DeinitializeActions)
    return OnInitialized(DeinitializeActions.takeError());

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Allocations[MinAddr].DeinitializationActions =
        std::move(*DeinitializeActions);
    Reservations[AI.MappingBase.toPtr<void *>()].Allocations.push_back(MinAddr);
  }

  OnInitialized(MinAddr);
}

void InProcessMemoryMapper::deinitialize(
    ArrayRef<ExecutorAddr> Bases,
    MemoryMapper::OnDeinitializedFunction OnDeinitialized) {
  Error AllErr = Error::success();

  {
    std::lock_guard<std::mutex> Lock(Mutex);

    for (auto Base : Bases) {

      if (Error Err = shared::runDeallocActions(
              Allocations[Base].DeinitializationActions)) {
        AllErr = joinErrors(std::move(AllErr), std::move(Err));
      }

      Allocations.erase(Base);
    }
  }

  OnDeinitialized(std::move(AllErr));
}

void InProcessMemoryMapper::release(ArrayRef<ExecutorAddr> Bases,
                                    OnReleasedFunction OnReleased) {
  Error Err = Error::success();

  for (auto Base : Bases) {
    std::vector<ExecutorAddr> AllocAddrs;
    size_t Size;
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      auto &R = Reservations[Base.toPtr<void *>()];
      Size = R.Size;
      AllocAddrs.swap(R.Allocations);
    }

    // deinitialize sub allocations
    std::promise<MSVCPError> P;
    auto F = P.get_future();
    deinitialize(AllocAddrs, [&](Error Err) { P.set_value(std::move(Err)); });
    if (Error E = F.get()) {
      Err = joinErrors(std::move(Err), std::move(E));
    }

    // free the memory
    auto MB = sys::MemoryBlock(Base.toPtr<void *>(), Size);

    auto EC = sys::Memory::releaseMappedMemory(MB);
    if (EC) {
      Err = joinErrors(std::move(Err), errorCodeToError(EC));
    }

    std::lock_guard<std::mutex> Lock(Mutex);
    Reservations.erase(Base.toPtr<void *>());
  }

  OnReleased(std::move(Err));
}

InProcessMemoryMapper::~InProcessMemoryMapper() {
  std::vector<ExecutorAddr> ReservationAddrs;
  {
    std::lock_guard<std::mutex> Lock(Mutex);

    ReservationAddrs.reserve(Reservations.size());
    for (const auto &R : Reservations) {
      ReservationAddrs.push_back(ExecutorAddr::fromPtr(R.getFirst()));
    }
  }

  std::promise<MSVCPError> P;
  auto F = P.get_future();
  release(ReservationAddrs, [&](Error Err) { P.set_value(std::move(Err)); });
  cantFail(F.get());
}

} // namespace orc

} // namespace llvm
