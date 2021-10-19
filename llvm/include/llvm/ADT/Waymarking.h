//===- Waymarking.h - Array waymarking algorithm ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility to backtrace an array's head, from a pointer into it. For the
// backtrace to work, we use "Waymarks", which are special tags embedded into
// the array's elements.
//
// A Tag of n-bits (in size) is composed as follows:
//
// bits: |   n-1   |             n-2 ... 0              |
//       .---------.------------------------------------.
//       |Stop Mask|(2^(n-1))-ary numeric system - digit|
//       '---------'------------------------------------'
//
// Backtracing is done as follows:
// Walk back (starting from a given pointer to an element into the array), until
// a tag with a "Stop Mask" is reached. Then start calculating the "Offset" from
// the array's head, by picking up digits along the way, until another stop is
// reached. The "Offset" is then subtracted from the current pointer, and the
// result is the array's head.
// A special case - if we first encounter a Tag with a Stop and a zero digit,
// then this is already the head.
//
// For example:
// In case of 2 bits:
//
// Tags:
// x0 - binary digit 0
// x1 - binary digit 1
// 1x - stop and calculate (s)
//
// Array:
//         .---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.
// head -> |s0 |s1 | 0 |s1 | 0 | 0 |s1 | 1 | 1 |s1 | 0 | 1 | 0 |s1 | 0 | 1 |
//         '---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'
//             |-1 |-2     |-4         |-7         |-10            |-14
//          <_ |   |       |           |           |               |
//          <_____ |       |           |           |               |
//          <_____________ |           |           |               |
//          <_________________________ |           |               |
//          <_____________________________________ |               |
//          <_____________________________________________________ |
//
//
// In case of 3 bits:
//
// Tags:
// x00 - quaternary digit 0
// x01 - quaternary digit 1
// x10 - quaternary digit 2
// x11 - quaternary digit 3
// 1xy - stop and calculate (s)
//
// Array:
//         .---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.
// head -> |s0 |s1 |s2 |s3 | 0 |s1 | 2 |s1 | 0 |s2 | 2 |s2 | 0 |s3 | 2 |s3 |
//         '---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'
//             |-1 |-2 |-3 |-4     |-6     |-8     |-10    |-12    |-14    |-16
//          <_ |   |   |   |       |       |       |       |       |       |
//          <_____ |   |   |       |       |       |       |       |       |
//          <_________ |   |       |       |       |       |       |       |
//          <_____________ |       |       |       |       |       |       |
//          <_____________________ |       |       |       |       |       |
//          <_____________________________ |       |       |       |       |
//          <_____________________________________ |       |       |       |
//          <_____________________________________________ |       |       |
//          <_____________________________________________________ |       |
//          <_____________________________________________________________ |
//
//
// The API introduce 2 functions:
// 1. fillWaymarks
// 2. followWaymarks
//
// Example:
//   int N = 10;
//   int M = 5;
//   int **A = new int *[N + M];   // Define the array.
//   for (int I = 0; I < N + M; ++I)
//     A[I] = new int(I);
//
//   fillWaymarks(A, A + N);       // Set the waymarks for the first N elements
//                                 // of the array.
//                                 // Note that it must be done AFTER we fill
//                                 // the array's elements.
//
//   ...                           // Elements which are not in the range
//                                 // [A, A+N) will not be marked, and we won't
//                                 // be able to call followWaymarks on them.
//
//   ...                           // Elements which will be changed after the
//                                 // call to fillWaymarks, will have to be
//                                 // retagged.
//
//   fillWaymarks(A + N, A + N + M, N); // Set the waymarks of the remaining M
//                                      // elements.
//   ...
//   int **It = A + N + 1;
//   int **B = followWaymarks(It); // Find the head of the array containing It.
//   assert(B == A);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_WAYMARKING_H
#define LLVM_ADT_WAYMARKING_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace llvm {

namespace detail {

template <unsigned NumBits> struct WaymarkingTraits {
  enum : unsigned {
    // The number of bits of a Waymarking Tag.
    NUM_BITS = NumBits,

    // A Tag is composed from a Mark and a Stop mask.
    MARK_SIZE = NUM_BITS - 1,
    STOP_MASK = (1 << MARK_SIZE),
    MARK_MASK = (STOP_MASK - 1),
    TAG_MASK = (MARK_MASK | STOP_MASK),

    // The number of pre-computed tags (for fast fill).
    NUM_STATIC_TAGS = 32
  };

private:
  // Add a new tag, calculated from Count and Stop, to the Vals pack, while
  // continuing recursively to decrease Len down to 0.
  template <unsigned Len, bool Stop, unsigned Count, uint8_t... Vals>
  struct AddTag;

  // Delegate to the specialized AddTag according to the need of a Stop mask.
  template <unsigned Len, unsigned Count, uint8_t... Vals> struct GenTag {
    typedef
        typename AddTag<Len, (Count <= MARK_MASK), Count, Vals...>::Xdata Xdata;
  };

  // Start adding tags while calculating the next Count, which is actually the
  // number of already calculated tags (equivalent to the position in the
  // array).
  template <unsigned Len, uint8_t... Vals> struct GenOffset {
    typedef typename GenTag<Len, sizeof...(Vals), Vals...>::Xdata Xdata;
  };

  // Add the tag and remove it from Count.
  template <unsigned Len, unsigned Count, uint8_t... Vals>
  struct AddTag<Len, false, Count, Vals...> {
    typedef typename GenTag<Len - 1, (Count >> MARK_SIZE), Vals...,
                            Count & MARK_MASK>::Xdata Xdata;
  };

  // We have reached the end of this Count, so start with a new Count.
  template <unsigned Len, unsigned Count, uint8_t... Vals>
  struct AddTag<Len, true, Count, Vals...> {
    typedef typename GenOffset<Len - 1, Vals...,
                               (Count & MARK_MASK) | STOP_MASK>::Xdata Xdata;
  };

  template <unsigned Count, uint8_t... Vals> struct TagsData {
    // The remaining number for calculating the next tag, following the last one
    // in Values.
    static const unsigned Remain = Count;

    // The array of ordered pre-computed Tags.
    static const uint8_t Values[sizeof...(Vals)];
  };

  // Specialize the case when Len equals 0, as the recursion stop condition.
  template <unsigned Count, uint8_t... Vals>
  struct AddTag<0, false, Count, Vals...> {
    typedef TagsData<Count, Vals...> Xdata;
  };

  template <unsigned Count, uint8_t... Vals>
  struct AddTag<0, true, Count, Vals...> {
    typedef TagsData<Count, Vals...> Xdata;
  };

public:
  typedef typename GenOffset<NUM_STATIC_TAGS>::Xdata Tags;
};

template <unsigned NumBits>
template <unsigned Count, uint8_t... Vals>
const uint8_t WaymarkingTraits<NumBits>::TagsData<
    Count, Vals...>::Values[sizeof...(Vals)] = {Vals...};

} // end namespace detail

/// This class is responsible for tagging (and retrieving the tag of) a given
/// element of type T.
template <class T, class WTraits = detail::WaymarkingTraits<
                       PointerLikeTypeTraits<T>::NumLowBitsAvailable>>
struct Waymarker {
  using Traits = WTraits;
  static void setWaymark(T &N, unsigned Tag) { N.setWaymark(Tag); }
  static unsigned getWaymark(const T &N) { return N.getWaymark(); }
};

template <class T, class WTraits> struct Waymarker<T *, WTraits> {
  using Traits = WTraits;
  static void setWaymark(T *&N, unsigned Tag) {
    reinterpret_cast<uintptr_t &>(N) |= static_cast<uintptr_t>(Tag);
  }
  static unsigned getWaymark(const T *N) {
    return static_cast<unsigned>(reinterpret_cast<uintptr_t>(N)) &
           Traits::TAG_MASK;
  }
};

/// Sets up the waymarking algorithm's tags for a given range [Begin, End).
///
/// \param Begin The beginning of the range to mark with tags (inclusive).
/// \param End The ending of the range to mark with tags (exclusive).
/// \param Offset The position in the supposed tags array from which to start
/// marking the given range.
template <class TIter, class Marker = Waymarker<
                           typename std::iterator_traits<TIter>::value_type>>
void fillWaymarks(TIter Begin, TIter End, size_t Offset = 0) {
  if (Begin == End)
    return;

  size_t Count = Marker::Traits::Tags::Remain;
  if (Offset <= Marker::Traits::NUM_STATIC_TAGS) {
    // Start by filling the pre-calculated tags, starting from the given offset.
    while (Offset != Marker::Traits::NUM_STATIC_TAGS) {
      Marker::setWaymark(*Begin, Marker::Traits::Tags::Values[Offset]);

      ++Offset;
      ++Begin;

      if (Begin == End)
        return;
    }
  } else {
    // The given offset is larger than the number of pre-computed tags, so we
    // must do it the hard way.
    // Calculate the next remaining Count, as if we have filled the tags up to
    // the given offset.
    size_t Off = Marker::Traits::NUM_STATIC_TAGS;
    do {
      ++Off;

      // If the count can fit into the tag, then the counting must stop.
      if (Count <= Marker::Traits::MARK_MASK) {
        Count = Off;
      } else
        Count >>= Marker::Traits::MARK_SIZE;
    } while (Off != Offset);
  }

  // By now, we have the matching remaining Count for the current offset.
  do {
    ++Offset;

    unsigned Tag = Count & Marker::Traits::MARK_MASK;

    // If the count can fit into the tag, then the counting must stop.
    if (Count <= Marker::Traits::MARK_MASK) {
      Tag |= Marker::Traits::STOP_MASK;
      Count = Offset;
    } else
      Count >>= Marker::Traits::MARK_SIZE;

    Marker::setWaymark(*Begin, Tag);
    ++Begin;
  } while (Begin != End);
}

/// Sets up the waymarking algorithm's tags for a given range.
///
/// \param Range The range to mark with tags.
/// \param Offset The position in the supposed tags array from which to start
/// marking the given range.
template <typename R, class Marker = Waymarker<typename std::remove_reference<
                          decltype(*std::begin(std::declval<R &>()))>::type>>
void fillWaymarks(R &&Range, size_t Offset = 0) {
  return fillWaymarks<decltype(std::begin(std::declval<R &>())), Marker>(
      adl_begin(Range), adl_end(Range), Offset);
}

/// Retrieves the element marked with tag of only STOP_MASK, by following the
/// waymarks. This is the first element in a range passed to a previous call to
/// \c fillWaymarks with \c Offset 0.
///
/// For the trivial usage of calling \c fillWaymarks(Array), and \I is an
/// iterator inside \c Array, this function retrieves the head of \c Array, by
/// following the waymarks.
///
/// \param I The iterator into an array which was marked by the waymarking tags
/// (by a previous call to \c fillWaymarks).
template <class TIter, class Marker = Waymarker<
                           typename std::iterator_traits<TIter>::value_type>>
TIter followWaymarks(TIter I) {
  unsigned Tag;
  do
    Tag = Marker::getWaymark(*I--);
  while (!(Tag & Marker::Traits::STOP_MASK));

  // Special case for the first Use.
  if (Tag != Marker::Traits::STOP_MASK) {
    ptrdiff_t Offset = Tag & Marker::Traits::MARK_MASK;
    while (!((Tag = Marker::getWaymark(*I)) & Marker::Traits::STOP_MASK)) {
      Offset = (Offset << Marker::Traits::MARK_SIZE) + Tag;
      --I;
    }
    I -= Offset;
  }
  return ++I;
}

} // end namespace llvm

#endif // LLVM_ADT_WAYMARKING_H
