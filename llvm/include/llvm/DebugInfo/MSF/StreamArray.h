//===- StreamArray.h - Array backed by an arbitrary stream ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_STREAMARRAY_H
#define LLVM_DEBUGINFO_MSF_STREAMARRAY_H

#include "llvm/DebugInfo/MSF/SequencedItemStream.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <type_traits>

namespace llvm {
namespace msf {

/// VarStreamArrayExtractor is intended to be specialized to provide customized
/// extraction logic.  On input it receives a StreamRef pointing to the
/// beginning of the next record, but where the length of the record is not yet
/// known.  Upon completion, it should return an appropriate Error instance if
/// a record could not be extracted, or if one could be extracted it should
/// return success and set Len to the number of bytes this record occupied in
/// the underlying stream, and it should fill out the fields of the value type
/// Item appropriately to represent the current record.
///
/// You can specialize this template for your own custom value types to avoid
/// having to specify a second template argument to VarStreamArray (documented
/// below).
template <typename T> struct VarStreamArrayExtractor {
  // Method intentionally deleted.  You must provide an explicit specialization
  // with the following method implemented.
  Error operator()(ReadableStreamRef Stream, uint32_t &Len,
                   T &Item) const = delete;
};

/// VarStreamArray represents an array of variable length records backed by a
/// stream.  This could be a contiguous sequence of bytes in memory, it could
/// be a file on disk, or it could be a PDB stream where bytes are stored as
/// discontiguous blocks in a file.  Usually it is desirable to treat arrays
/// as contiguous blocks of memory, but doing so with large PDB files, for
/// example, could mean allocating huge amounts of memory just to allow
/// re-ordering of stream data to be contiguous before iterating over it.  By
/// abstracting this out, we need not duplicate this memory, and we can
/// iterate over arrays in arbitrarily formatted streams.  Elements are parsed
/// lazily on iteration, so there is no upfront cost associated with building
/// a VarStreamArray, no matter how large it may be.
///
/// You create a VarStreamArray by specifying a ValueType and an Extractor type.
/// If you do not specify an Extractor type, it expects you to specialize
/// VarStreamArrayExtractor<T> for your ValueType.
///
/// By default an Extractor is default constructed in the class, but in some
/// cases you might find it useful for an Extractor to maintain state across
/// extractions.  In this case you can provide your own Extractor through a
/// secondary constructor.  The following examples show various ways of
/// creating a VarStreamArray.
///
///       // Will use VarStreamArrayExtractor<MyType> as the extractor.
///       VarStreamArray<MyType> MyTypeArray;
///
///       // Will use a default-constructed MyExtractor as the extractor.
///       VarStreamArray<MyType, MyExtractor> MyTypeArray2;
///
///       // Will use the specific instance of MyExtractor provided.
///       // MyExtractor need not be default-constructible in this case.
///       MyExtractor E(SomeContext);
///       VarStreamArray<MyType, MyExtractor> MyTypeArray3(E);
///
template <typename ValueType, typename Extractor> class VarStreamArrayIterator;

template <typename ValueType,
          typename Extractor = VarStreamArrayExtractor<ValueType>>
class VarStreamArray {
  friend class VarStreamArrayIterator<ValueType, Extractor>;

public:
  typedef VarStreamArrayIterator<ValueType, Extractor> Iterator;

  VarStreamArray() {}
  explicit VarStreamArray(const Extractor &E) : E(E) {}

  explicit VarStreamArray(ReadableStreamRef Stream) : Stream(Stream) {}
  VarStreamArray(ReadableStreamRef Stream, const Extractor &E)
      : Stream(Stream), E(E) {}

  VarStreamArray(const VarStreamArray<ValueType, Extractor> &Other)
      : Stream(Other.Stream), E(Other.E) {}

  Iterator begin(bool *HadError = nullptr) const {
    return Iterator(*this, E, HadError);
  }

  Iterator end() const { return Iterator(E); }

  const Extractor &getExtractor() const { return E; }

  ReadableStreamRef getUnderlyingStream() const { return Stream; }

private:
  ReadableStreamRef Stream;
  Extractor E;
};

template <typename ValueType, typename Extractor> class VarStreamArrayIterator {
  typedef VarStreamArrayIterator<ValueType, Extractor> IterType;
  typedef VarStreamArray<ValueType, Extractor> ArrayType;

public:
  VarStreamArrayIterator(const ArrayType &Array, const Extractor &E,
                         bool *HadError = nullptr)
      : IterRef(Array.Stream), Array(&Array), HadError(HadError), Extract(E) {
    if (IterRef.getLength() == 0)
      moveToEnd();
    else {
      auto EC = Extract(IterRef, ThisLen, ThisValue);
      if (EC) {
        consumeError(std::move(EC));
        markError();
      }
    }
  }
  VarStreamArrayIterator() {}
  explicit VarStreamArrayIterator(const Extractor &E) : Extract(E) {}
  ~VarStreamArrayIterator() {}

  bool operator==(const IterType &R) const {
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

  bool operator!=(const IterType &R) { return !(*this == R); }

  const ValueType &operator*() const {
    assert(Array && !HasError);
    return ThisValue;
  }

  IterType &operator++() {
    // We are done with the current record, discard it so that we are
    // positioned at the next record.
    IterRef = IterRef.drop_front(ThisLen);
    if (IterRef.getLength() == 0) {
      // There is nothing after the current record, we must make this an end
      // iterator.
      moveToEnd();
    } else {
      // There is some data after the current record.
      auto EC = Extract(IterRef, ThisLen, ThisValue);
      if (EC) {
        consumeError(std::move(EC));
        markError();
      } else if (ThisLen == 0) {
        // An empty record? Make this an end iterator.
        moveToEnd();
      }
    }
    return *this;
  }

  IterType operator++(int) {
    IterType Original = *this;
    ++*this;
    return Original;
  }

private:
  void moveToEnd() {
    Array = nullptr;
    ThisLen = 0;
  }
  void markError() {
    moveToEnd();
    HasError = true;
    if (HadError != nullptr)
      *HadError = true;
  }

  ValueType ThisValue;
  ReadableStreamRef IterRef;
  const ArrayType *Array{nullptr};
  uint32_t ThisLen{0};
  bool HasError{false};
  bool *HadError{nullptr};
  Extractor Extract;
};

template <typename T> class FixedStreamArrayIterator;

template <typename T> class FixedStreamArray {
  friend class FixedStreamArrayIterator<T>;

public:
  FixedStreamArray() : Stream() {}
  FixedStreamArray(ReadableStreamRef Stream) : Stream(Stream) {
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
    return FixedStreamArrayIterator<T>(*this, size());
  }

  ReadableStreamRef getUnderlyingStream() const { return Stream; }

private:
  ReadableStreamRef Stream;
};

template <typename T> class FixedStreamArrayIterator {
public:
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
    assert(Index < Array.size());
    ++Index;
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

} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMARRAY_H
