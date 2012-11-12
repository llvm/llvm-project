//===-- llvm/ADT/edit_distance.h - Array edit distance function --- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a Levenshtein distance function that works for any two
// sequences, with each element of each sequence being analogous to a character
// in a string.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_EDIT_DISTANCE_H
#define LLVM_ADT_EDIT_DISTANCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include <algorithm>

namespace llvm {

/// \brief Determine the edit distance between two sequences.
///
/// \param FromArray the first sequence to compare.
///
/// \param ToArray the second sequence to compare.
///
/// \param AllowReplacements whether to allow element replacements (change one
/// element into another) as a single operation, rather than as two operations
/// (an insertion and a removal).
///
/// \param MaxEditDistance If non-zero, the maximum edit distance that this
/// routine is allowed to compute. If the edit distance will exceed that
/// maximum, returns \c MaxEditDistance+1.
///
/// \returns the minimum number of element insertions, removals, or (if
/// \p AllowReplacements is \c true) replacements needed to transform one of
/// the given sequences into the other. If zero, the sequences are identical.
template<typename T>
unsigned ComputeEditDistance(ArrayRef<T> FromArray, ArrayRef<T> ToArray,
                             bool AllowReplacements = true,
                             unsigned MaxEditDistance = 0) {
  // The algorithm implemented below is the "classic"
  // dynamic-programming algorithm for computing the Levenshtein
  // distance, which is described here:
  //
  //   http://en.wikipedia.org/wiki/Levenshtein_distance
  //
  // Although the algorithm is typically described using an m x n
  // array, only two rows are used at a time, so this implemenation
  // just keeps two separate vectors for those two rows.
  typename ArrayRef<T>::size_type m = FromArray.size();
  typename ArrayRef<T>::size_type n = ToArray.size();

  const unsigned SmallBufferSize = 64;
  unsigned SmallBuffer[SmallBufferSize];
  llvm::OwningArrayPtr<unsigned> Allocated;
  unsigned *Previous = SmallBuffer;
  if (2*(n + 1) > SmallBufferSize) {
    Previous = new unsigned [2*(n+1)];
    Allocated.reset(Previous);
  }
  unsigned *Current = Previous + (n + 1);

  for (unsigned i = 0; i <= n; ++i)
    Previous[i] = i;

  for (typename ArrayRef<T>::size_type y = 1; y <= m; ++y) {
    Current[0] = y;
    unsigned BestThisRow = Current[0];

    for (typename ArrayRef<T>::size_type x = 1; x <= n; ++x) {
      if (AllowReplacements) {
        Current[x] = std::min(
            Previous[x-1] + (FromArray[y-1] == ToArray[x-1] ? 0u : 1u),
            std::min(Current[x-1], Previous[x])+1);
      }
      else {
        if (FromArray[y-1] == ToArray[x-1]) Current[x] = Previous[x-1];
        else Current[x] = std::min(Current[x-1], Previous[x]) + 1;
      }
      BestThisRow = std::min(BestThisRow, Current[x]);
    }

    if (MaxEditDistance && BestThisRow > MaxEditDistance)
      return MaxEditDistance + 1;

    unsigned *tmp = Current;
    Current = Previous;
    Previous = tmp;
  }

  unsigned Result = Previous[n];
  return Result;
}

} // End llvm namespace

#endif
