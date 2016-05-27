//===- StreamArray.h - Array backed by an arbitrary stream ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_STREAMARRAY_H
#define LLVM_DEBUGINFO_CODEVIEW_STREAMARRAY_H

#include "llvm/DebugInfo/CodeView/StreamRef.h"

#include <functional>
#include <type_traits>

namespace llvm {
namespace codeview {

/// VarStreamArray represents an array of variable length records backed by a
/// stream.  This could be a contiguous sequence of bytes in memory, it could
/// be a file on disk, or it could be a PDB stream where bytes are stored as
/// discontiguous blocks in a file.  Usually it is desirable to treat arrays
/// as contiguous blocks of memory, but doing so with large PDB files, for
/// example, could mean allocating huge amounts of memory just to allow
/// re-ordering of stream data to be contiguous before iterating over it.  By
/// abstracting this out, we need not duplicate this memory, and we can
/// iterate over arrays in arbitrarily formatted streams.
class VarStreamArrayIterator;

class VarStreamArray {
  friend class VarStreamArrayIterator;
  typedef std::function<uint32_t(const StreamInterface &)> LengthFuncType;

public:
  template <typename LengthFunc>
  VarStreamArray(StreamRef Stream, const LengthFunc &Len)
      : Stream(Stream), Len(Len) {}

  VarStreamArrayIterator begin() const;
  VarStreamArrayIterator end() const;

private:
  StreamRef Stream;
  LengthFuncType Len; // Function used to calculate legth of a record
};

class VarStreamArrayIterator {
public:
  VarStreamArrayIterator(const VarStreamArray &Array)
      : Array(&Array), IterRef(Array.Stream) {
    ThisLen = Array.Len(IterRef);
  }
  VarStreamArrayIterator() : Array(nullptr), IterRef() {}
  bool operator==(const VarStreamArrayIterator &R) const {
    if (Array && R.Array) {
      // Both have a valid array, make sure they're same.
      assert(Array == R.Array);
      return IterRef == R.IterRef;
    }

    // Both iterators are at the end.
    if (!Array && !R.Array)
      return true;

    // One is not at the end and one is.
    return false;
  }

  bool operator!=(const VarStreamArrayIterator &R) { return !(*this == R); }

  StreamRef operator*() const {
    ArrayRef<uint8_t> Result;
    return IterRef.keep_front(ThisLen);
  }

  VarStreamArrayIterator &operator++() {
    if (!Array || IterRef.getLength() == 0)
      return *this;
    IterRef = IterRef.drop_front(ThisLen);
    if (IterRef.getLength() == 0) {
      Array = nullptr;
      ThisLen = 0;
    } else {
      ThisLen = Array->Len(IterRef);
    }
    return *this;
  }

  VarStreamArrayIterator operator++(int) {
    VarStreamArrayIterator Original = *this;
    ++*this;
    return Original;
  }

private:
  const VarStreamArray *Array;
  uint32_t ThisLen;
  StreamRef IterRef;
};

inline VarStreamArrayIterator VarStreamArray::begin() const {
  return VarStreamArrayIterator(*this);
}
inline VarStreamArrayIterator VarStreamArray::end() const {
  return VarStreamArrayIterator();
}

template <typename T> class FixedStreamArrayIterator;

template <typename T> class FixedStreamArray {
  friend class FixedStreamArrayIterator<T>;

public:
  FixedStreamArray() : Stream() {}
  FixedStreamArray(StreamRef Stream) : Stream(Stream) {
    assert(Stream.getLength() % sizeof(T) == 0);
  }

  const T &operator[](uint32_t Index) const {
    assert(Index < size());
    uint32_t Off = Index * sizeof(T);
    ArrayRef<uint8_t> Data;
    if (auto EC = Stream.readBytes(Off, sizeof(T), Data)) {
      assert(false && "Unexpected failure reading from stream");
      // This should never happen since we asserted that the stream length was
      // an exact multiple of the element size.
      consumeError(std::move(EC));
    }
    return *reinterpret_cast<const T *>(Data.data());
  }

  uint32_t size() const { return Stream.getLength() / sizeof(T); }

  FixedStreamArrayIterator<T> begin() const {
    return FixedStreamArrayIterator<T>(*this, 0);
  }
  FixedStreamArrayIterator<T> end() const {
    return FixedStreamArrayIterator<T>(*this);
  }

private:
  StreamRef Stream;
};

template <typename T> class FixedStreamArrayIterator {
public:
  FixedStreamArrayIterator(const FixedStreamArray<T> &Array)
      : Array(Array), Index(uint32_t(-1)) {}
  FixedStreamArrayIterator(const FixedStreamArray<T> &Array, uint32_t Index)
      : Array(Array), Index(Index) {}

  bool operator==(const FixedStreamArrayIterator<T> &R) {
    assert(&Array == &R.Array);
    return Index == R.Index;
  }

  bool operator!=(const FixedStreamArrayIterator<T> &R) {
    return !(*this == R);
  }

  const T &operator*() const { return Array[Index]; }

  FixedStreamArrayIterator<T> &operator++() {
    if (Index == uint32_t(-1))
      return *this;
    if (++Index >= Array.size())
      Index = uint32_t(-1);
    return *this;
  }

  FixedStreamArrayIterator<T> operator++(int) {
    FixedStreamArrayIterator<T> Original = *this;
    ++*this;
    return Original;
  }

private:
  const FixedStreamArray<T> &Array;
  uint32_t Index;
};

} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_STREAMARRAY_H
