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
class APSInt;

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

/// Decodes a numeric "leaf" value. These are integer literals encountered in
/// the type stream. If the value is positive and less than LF_NUMERIC (1 <<
/// 15), it is emitted directly in Data. Otherwise, it has a tag like LF_CHAR
/// that indicates the bitwidth and sign of the numeric data.
bool decodeNumericLeaf(ArrayRef<uint8_t> &Data, APSInt &Num);

inline bool decodeNumericLeaf(StringRef &Data, APSInt &Num) {
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(Data.data()),
                          Data.size());
  bool Success = decodeNumericLeaf(Bytes, Num);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return Success;
}

/// Decode a numeric leaf value that is known to be a uint32_t.
bool decodeUIntLeaf(ArrayRef<uint8_t> &Data, uint64_t &Num);

// A const input iterator interface to the CodeView type stream.
class TypeIterator {
public:
  struct TypeRecord {
    std::size_t Length;
    TypeLeafKind Leaf;
    ArrayRef<uint8_t> LeafData;
  };

  explicit TypeIterator(const ArrayRef<uint8_t> &SectionData)
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

    // FIXME: Use consumeObject when it deals in ArrayRef<uint8_t>.
    if (Data.size() < sizeof(TypeRecordPrefix))
      return;
    const auto *Rec = reinterpret_cast<const TypeRecordPrefix *>(Data.data());
    Data = Data.drop_front(sizeof(TypeRecordPrefix));

    Current.Length = Rec->Len;
    Current.Leaf = static_cast<TypeLeafKind>(uint16_t(Rec->Leaf));
    Current.LeafData = Data.slice(0, Current.Length - 2);

    // The next record starts immediately after this one.
    Data = Data.drop_front(Current.LeafData.size());

    // FIXME: The stream contains LF_PAD bytes that we need to ignore, but those
    // are typically included in LeafData. We may need to call skipPadding() if
    // we ever find a record that doesn't count those bytes.

    return;
  }

  ArrayRef<uint8_t> Data;
  TypeRecord Current;
  bool AtEnd;
};

inline iterator_range<TypeIterator> makeTypeRange(ArrayRef<uint8_t> Data) {
  return make_range(TypeIterator(Data), TypeIterator());
}

} // end namespace codeview
} // end namespace llvm

#endif
