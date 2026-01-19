//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares interface for OnDiskDataAllocator, a file backed data
/// pool can be used to allocate space to store data packed in a single file. It
/// is based on MappedFileRegionArena and includes a header in the beginning to
/// provide metadata.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKDATAALLOCATOR_H
#define LLVM_CAS_ONDISKDATAALLOCATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CAS/FileOffset.h"
#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/Support/Error.h"

namespace llvm::cas {

/// Sink for data. Stores variable length data with 8-byte alignment. Does not
/// track size of data, which is assumed to known from context, or embedded.
/// Uses 0-padding but does not guarantee 0-termination.
class OnDiskDataAllocator {
public:
  using ValueProxy = MutableArrayRef<char>;

  /// A pointer to data stored on disk.
  class OnDiskPtr {
  public:
    FileOffset getOffset() const { return Offset; }
    explicit operator bool() const { return bool(getOffset()); }
    const ValueProxy &operator*() const {
      assert(Offset && "Null dereference");
      return Value;
    }
    const ValueProxy *operator->() const {
      assert(Offset && "Null dereference");
      return &Value;
    }

    OnDiskPtr() = default;

  private:
    friend class OnDiskDataAllocator;
    OnDiskPtr(FileOffset Offset, ValueProxy Value)
        : Offset(Offset), Value(Value) {}
    FileOffset Offset;
    ValueProxy Value;
  };

  /// Get the data of \p Size stored at the given \p Offset. Note the allocator
  /// doesn't keep track of the allocation size, thus \p Size doesn't need to
  /// match the size of allocation but needs to be smaller.
  LLVM_ABI_FOR_TEST Expected<ArrayRef<char>> get(FileOffset Offset,
                                                 size_t Size) const;

  /// Allocate at least \p Size with 8-byte alignment.
  LLVM_ABI_FOR_TEST Expected<OnDiskPtr> allocate(size_t Size);

  /// \returns the buffer that was allocated at \p create time, with size
  /// \p UserHeaderSize.
  MutableArrayRef<uint8_t> getUserHeader() const;

  LLVM_ABI_FOR_TEST size_t size() const;
  LLVM_ABI_FOR_TEST size_t capacity() const;

  LLVM_ABI_FOR_TEST static Expected<OnDiskDataAllocator>
  create(const Twine &Path, const Twine &TableName, uint64_t MaxFileSize,
         std::optional<uint64_t> NewFileInitialSize,
         uint32_t UserHeaderSize = 0,
         std::shared_ptr<ondisk::OnDiskCASLogger> Logger = nullptr,
         function_ref<void(void *)> UserHeaderInit = nullptr);

  LLVM_ABI_FOR_TEST OnDiskDataAllocator(OnDiskDataAllocator &&RHS);
  LLVM_ABI_FOR_TEST OnDiskDataAllocator &operator=(OnDiskDataAllocator &&RHS);

  // No copy. Just call \a create() again.
  OnDiskDataAllocator(const OnDiskDataAllocator &) = delete;
  OnDiskDataAllocator &operator=(const OnDiskDataAllocator &) = delete;

  LLVM_ABI_FOR_TEST ~OnDiskDataAllocator();

private:
  struct ImplType;
  explicit OnDiskDataAllocator(std::unique_ptr<ImplType> Impl);
  std::unique_ptr<ImplType> Impl;
};

} // namespace llvm::cas

#endif // LLVM_CAS_ONDISKDATAALLOCATOR_H
