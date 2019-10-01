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
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"
#include <cstdint>

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
    SegmentRequest(size_t ContentSize, unsigned ContentAlign,
                   uint64_t ZeroFillSize, unsigned ZeroFillAlign)
        : ContentSize(ContentSize), ZeroFillSize(ZeroFillSize),
          ContentAlign(ContentAlign), ZeroFillAlign(ZeroFillAlign) {}
    size_t getContentSize() const { return ContentSize; }
    unsigned getContentAlignment() const { return ContentAlign; }
    uint64_t getZeroFillSize() const { return ZeroFillSize; }
    unsigned getZeroFillAlignment() const { return ZeroFillAlign; }

  private:
    size_t ContentSize = 0;
    uint64_t ZeroFillSize = 0;
    unsigned ContentAlign = 0;
    unsigned ZeroFillAlign = 0;
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

    /// Should deallocate target memory.
    virtual Error deallocate() = 0;
  };

  virtual ~JITLinkMemoryManager();

  /// Create an Allocation object.
  virtual Expected<std::unique_ptr<Allocation>>
  allocate(const SegmentsRequestMap &Request) = 0;
};

/// A JITLinkMemoryManager that allocates in-process memory.
class InProcessMemoryManager : public JITLinkMemoryManager {
public:
  Expected<std::unique_ptr<Allocation>>
  allocate(const SegmentsRequestMap &Request) override;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINK_H
