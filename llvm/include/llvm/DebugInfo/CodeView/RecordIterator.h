//===- RecordIterator.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_RECORDITERATOR_H
#define LLVM_DEBUGINFO_CODEVIEW_RECORDITERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {
// A const input iterator interface to the CodeView record stream.
template <typename Kind> class RecordIterator {
private:
  struct RecordPrefix {
    support::ulittle16_t RecordLen;  // Record length, starting from &Leaf.
    support::ulittle16_t RecordKind; // Record kind (from the `Kind` enum).
  };

public:
  struct Record {
    std::size_t Length;
    Kind Type;
    ArrayRef<uint8_t> Data;
  };

  explicit RecordIterator(const ArrayRef<uint8_t> &RecordBytes, bool *HadError)
      : HadError(HadError), Data(RecordBytes), AtEnd(false) {
    next(); // Prime the pump
  }

  RecordIterator() : HadError(nullptr), AtEnd(true) {}

  // For iterators to compare equal, they must both point at the same record
  // in the same data stream, or they must both be at the end of a stream.
  friend bool operator==(const RecordIterator<Kind> &lhs,
                         const RecordIterator<Kind> &rhs) {
    return (lhs.Data.begin() == rhs.Data.begin()) || (lhs.AtEnd && rhs.AtEnd);
  }

  friend bool operator!=(const RecordIterator<Kind> &lhs,
                         const RecordIterator<Kind> &rhs) {
    return !(lhs == rhs);
  }

  const Record &operator*() const {
    assert(!AtEnd);
    return Current;
  }

  const Record *operator->() const {
    assert(!AtEnd);
    return &Current;
  }

  RecordIterator<Kind> &operator++() {
    next();
    return *this;
  }

  RecordIterator<Kind> operator++(int) {
    RecordIterator<Kind> Original = *this;
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
    if (Data.size() < sizeof(RecordPrefix))
      return parseError();
    const auto *Rec = reinterpret_cast<const RecordPrefix *>(Data.data());
    Data = Data.drop_front(sizeof(RecordPrefix));

    Current.Length = Rec->RecordLen;
    Current.Type = static_cast<Kind>(uint16_t(Rec->RecordKind));
    size_t RecLen = Current.Length - 2;
    if (RecLen > Data.size())
      return parseError();
    Current.Data = Data.slice(0, RecLen);

    // The next record starts immediately after this one.
    Data = Data.drop_front(Current.Data.size());

    // FIXME: The stream contains LF_PAD bytes that we need to ignore, but those
    // are typically included in LeafData. We may need to call skipPadding() if
    // we ever find a record that doesn't count those bytes.

    return;
  }

  void parseError() {
    if (HadError)
      *HadError = true;
  }

  bool *HadError;
  ArrayRef<uint8_t> Data;
  Record Current;
  bool AtEnd;
};

template <typename Kind>
inline iterator_range<RecordIterator<Kind>>
makeRecordRange(ArrayRef<uint8_t> Data, bool *HadError) {
  return make_range(RecordIterator<Kind>(Data, HadError), RecordIterator<Kind>());
}
}
}

#endif
