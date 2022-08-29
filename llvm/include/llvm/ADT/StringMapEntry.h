//===- StringMapEntry.h - String Hash table map interface -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the StringMapEntry class - it is intended to be a low
/// dependency implementation detail of StringMap that is more suitable for
/// inclusion in public headers than StringMap.h itself is.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGMAPENTRY_H
#define LLVM_ADT_STRINGMAPENTRY_H

#include "llvm/ADT/None.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// StringMapEntryBase - Shared base class of StringMapEntry instances.
class StringMapEntryBase {
  size_t keyLength;

public:
  explicit StringMapEntryBase(size_t keyLength) : keyLength(keyLength) {}

  size_t getKeyLength() const { return keyLength; }

protected:
  /// Helper to tail-allocate \p Key. It'd be nice to generalize this so it
  /// could be reused elsewhere, maybe even taking an llvm::function_ref to
  /// type-erase the allocator and put it in a source file.
  template <typename AllocatorTy>
  static void *allocateWithKey(size_t EntrySize, size_t EntryAlign,
                               StringRef Key, AllocatorTy &Allocator);
};

// Define out-of-line to dissuade inlining.
template <typename AllocatorTy>
void *StringMapEntryBase::allocateWithKey(size_t EntrySize, size_t EntryAlign,
                                          StringRef Key,
                                          AllocatorTy &Allocator) {
  size_t KeyLength = Key.size();

  // Allocate a new item with space for the string at the end and a null
  // terminator.
  size_t AllocSize = EntrySize + KeyLength + 1;
  void *Allocation = Allocator.Allocate(AllocSize, EntryAlign);
  assert(Allocation && "Unhandled out-of-memory");

  // Copy the string information.
  char *Buffer = reinterpret_cast<char *>(Allocation) + EntrySize;
  if (KeyLength > 0)
    ::memcpy(Buffer, Key.data(), KeyLength);
  Buffer[KeyLength] = 0; // Null terminate for convenience of clients.
  return Allocation;
}

/// StringMapEntryStorage - Holds the value in a StringMapEntry.
///
/// Factored out into a separate base class to make it easier to specialize.
/// This is primarily intended to support StringSet, which doesn't need a value
/// stored at all.
template <typename ValueTy>
class StringMapEntryStorage : public StringMapEntryBase {
public:
  ValueTy second;

  explicit StringMapEntryStorage(size_t keyLength)
      : StringMapEntryBase(keyLength), second() {}
  template <typename... InitTy>
  StringMapEntryStorage(size_t keyLength, InitTy &&...initVals)
      : StringMapEntryBase(keyLength),
        second(std::forward<InitTy>(initVals)...) {}
  StringMapEntryStorage(StringMapEntryStorage &e) = delete;

  const ValueTy &getValue() const { return second; }
  ValueTy &getValue() { return second; }

  void setValue(const ValueTy &V) { second = V; }
};

template <> class StringMapEntryStorage<NoneType> : public StringMapEntryBase {
public:
  explicit StringMapEntryStorage(size_t keyLength, NoneType = None)
      : StringMapEntryBase(keyLength) {}
  StringMapEntryStorage(StringMapEntryStorage &entry) = delete;

  NoneType getValue() const { return None; }
};

/// StringMapEntry - This is used to represent one value that is inserted into
/// a StringMap.  It contains the Value itself and the key: the string length
/// and data.
template <typename ValueTy>
class StringMapEntry final : public StringMapEntryStorage<ValueTy> {
public:
  using StringMapEntryStorage<ValueTy>::StringMapEntryStorage;

  StringRef getKey() const {
    return StringRef(getKeyData(), this->getKeyLength());
  }

  /// getKeyData - Return the start of the string data that is the key for this
  /// value.  The string data is always stored immediately after the
  /// StringMapEntry object.
  const char *getKeyData() const {
    return reinterpret_cast<const char *>(this + 1);
  }

  StringRef first() const {
    return StringRef(getKeyData(), this->getKeyLength());
  }

  /// Create a StringMapEntry for the specified key construct the value using
  /// \p InitiVals.
  template <typename AllocatorTy, typename... InitTy>
  static StringMapEntry *Create(StringRef key, AllocatorTy &allocator,
                                InitTy &&...initVals) {
    return new (StringMapEntryBase::allocateWithKey(
        sizeof(StringMapEntry), alignof(StringMapEntry), key, allocator))
        StringMapEntry(key.size(), std::forward<InitTy>(initVals)...);
  }

  /// GetStringMapEntryFromKeyData - Given key data that is known to be embedded
  /// into a StringMapEntry, return the StringMapEntry itself.
  static StringMapEntry &GetStringMapEntryFromKeyData(const char *keyData) {
    char *ptr = const_cast<char *>(keyData) - sizeof(StringMapEntry<ValueTy>);
    return *reinterpret_cast<StringMapEntry *>(ptr);
  }

  /// Destroy - Destroy this StringMapEntry, releasing memory back to the
  /// specified allocator.
  template <typename AllocatorTy> void Destroy(AllocatorTy &allocator) {
    // Free memory referenced by the item.
    size_t AllocSize = sizeof(StringMapEntry) + this->getKeyLength() + 1;
    this->~StringMapEntry();
    allocator.Deallocate(static_cast<void *>(this), AllocSize,
                         alignof(StringMapEntry));
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGMAPENTRY_H
