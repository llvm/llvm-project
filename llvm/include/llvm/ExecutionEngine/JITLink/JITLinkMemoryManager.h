//===-- JITLinkMemoryManager.h - JITLink mem manager interface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the JITLinkMemoryManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_JITLINKMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_JITLINK_JITLINKMEMORYMANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Support/Memory.h"

#include <cstdint>
#include <future>

namespace llvm {
namespace jitlink {

/// Manages allocations of JIT memory.
///
/// Instances of this class may be accessed concurrently from multiple threads
/// and their implemetations should include any necessary synchronization.
class JITLinkMemoryManager {
public:
  using ProtectionFlags = sys::Memory::ProtectionFlags;

  class SegmentRequest {
  public:
    SegmentRequest() = default;
    SegmentRequest(uint64_t Alignment, size_t ContentSize,
                   uint64_t ZeroFillSize)
        : Alignment(Alignment), ContentSize(ContentSize),
          ZeroFillSize(ZeroFillSize) {
      assert(isPowerOf2_32(Alignment) && "Alignment must be power of 2");
    }
    uint64_t getAlignment() const { return Alignment; }
    size_t getContentSize() const { return ContentSize; }
    uint64_t getZeroFillSize() const { return ZeroFillSize; }
  private:
    uint64_t Alignment = 0;
    size_t ContentSize = 0;
    uint64_t ZeroFillSize = 0;
  };

  using SegmentsRequestMap = DenseMap<unsigned, SegmentRequest>;

  /// Represents an allocation created by the memory manager.
  ///
  /// An allocation object is responsible for allocating and owning jit-linker
  /// working and target memory, and for transfering from working to target
  /// memory.
  ///
  class Allocation {
  public:
    using FinalizeContinuation = std::function<void(Error)>;

    virtual ~Allocation();

    /// Should return the address of linker working memory for the segment with
    /// the given protection flags.
    virtual MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) = 0;

    /// Should return the final address in the target process where the segment
    /// will reside.
    virtual JITTargetAddress getTargetMemory(ProtectionFlags Seg) = 0;

    /// Should transfer from working memory to target memory, and release
    /// working memory.
    virtual void finalizeAsync(FinalizeContinuation OnFinalize) = 0;

    /// Calls finalizeAsync and waits for completion.
    Error finalize() {
      std::promise<MSVCPError> FinalizeResultP;
      auto FinalizeResultF = FinalizeResultP.get_future();
      finalizeAsync(
          [&](Error Err) { FinalizeResultP.set_value(std::move(Err)); });
      return FinalizeResultF.get();
    }

    /// Should deallocate target memory.
    virtual Error deallocate() = 0;
  };

  virtual ~JITLinkMemoryManager();

  /// Create an Allocation object.
  ///
  /// The JD argument represents the target JITLinkDylib, and can be used by
  /// JITLinkMemoryManager implementers to manage per-dylib allocation pools
  /// (e.g. one pre-reserved address space slab per dylib to ensure that all
  /// allocations for the dylib are within a certain range). The JD argument
  /// may be null (representing an allocation not associated with any
  /// JITDylib.
  ///
  /// The request argument describes the segment sizes and permisssions being
  /// requested.
  virtual Expected<std::unique_ptr<Allocation>>
  allocate(const JITLinkDylib *JD, const SegmentsRequestMap &Request) = 0;
};

/// A JITLinkMemoryManager that allocates in-process memory.
class InProcessMemoryManager : public JITLinkMemoryManager {
public:
  Expected<std::unique_ptr<Allocation>>
  allocate(const JITLinkDylib *JD, const SegmentsRequestMap &Request) override;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINKMEMORYMANAGER_H
