//===- TypeStream.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESTREAM_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESTREAM_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"

#include <stdint.h>

namespace llvm {
namespace codeview {

/// Consumes sizeof(T) bytes from the given byte sequence. Returns an error if
/// there are not enough bytes remaining. Reinterprets the consumed bytes as a
/// T object and points 'Res' at them.
template <typename T>
inline std::error_code consumeObject(StringRef &Data, const T *&Res) {
  if (Data.size() < sizeof(*Res))
    return object::object_error::parse_failed;
  Res = reinterpret_cast<const T *>(Data.data());
  Data = Data.drop_front(sizeof(*Res));
  return std::error_code();
}

inline std::error_code consumeUInt32(StringRef &Data, uint32_t &Res) {
  const support::ulittle32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Res = *IntPtr;
  return std::error_code();
}

// A const input iterator interface to the CodeView type stream.
class TypeIterator {
public:
  struct TypeRecord {
    std::size_t Length;
    TypeLeafKind Leaf;
    StringRef LeafData;
  };

  explicit TypeIterator(const StringRef &SectionData)
      : Data(SectionData), AtEnd(false) {
    next(); // Prime the pump
  }

  TypeIterator() : AtEnd(true) {}

  // For iterators to compare equal, they must both point at the same record
  // in the same data stream, or they must both be at the end of a stream.
  friend bool operator==(const TypeIterator &lhs, const TypeIterator &rhs) {
    return (lhs.Data.begin() == rhs.Data.begin()) || (lhs.AtEnd && rhs.AtEnd);
  }

  friend bool operator!=(const TypeIterator &lhs, const TypeIterator &rhs) {
    return !(lhs == rhs);
  }

  const TypeRecord &operator*() const {
    assert(!AtEnd);
    return Current;
  }

  const TypeRecord *operator->() const {
    assert(!AtEnd);
    return &Current;
  }

  TypeIterator operator++() {
    next();
    return *this;
  }

  TypeIterator operator++(int) {
    TypeIterator Original = *this;
    ++*this;
    return Original;
  }

private:
  void next() {
    assert(!AtEnd && "Attempted to advance more than one past the last rec");
    if (Data.empty()) {
      // We've advanced past the last record.
      AtEnd = true;
      return;
    }

    const TypeRecordPrefix *Rec;
    if (consumeObject(Data, Rec))
      return;
    Current.Length = Rec->Len;
    Current.Leaf = static_cast<TypeLeafKind>(uint16_t(Rec->Leaf));
    Current.LeafData = Data.substr(0, Current.Length - 2);

    // The next record starts immediately after this one.
    Data = Data.drop_front(Current.LeafData.size());

    // FIXME: The stream contains LF_PAD bytes that we need to ignore, but those
    // are typically included in LeafData. We may need to call skipPadding() if
    // we ever find a record that doesn't count those bytes.

    return;
  }

  StringRef Data;
  TypeRecord Current;
  bool AtEnd;
};

inline iterator_range<TypeIterator> makeTypeRange(StringRef Data) {
  return make_range(TypeIterator(Data), TypeIterator());
}
}
}

#endif
