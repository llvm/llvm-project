//===- LongestCommonSequence.h - Compute LCS --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements longestCommonSequence, useful for finding matches
// between two sequences, such as lists of profiling points.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LONGESTCOMMONSEQEUNCE_H
#define LLVM_TRANSFORMS_UTILS_LONGESTCOMMONSEQEUNCE_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <vector>

namespace llvm {

// This function implements the Myers diff algorithm used for stale profile
// matching. The algorithm provides a simple and efficient way to find the
// Longest Common Subsequence(LCS) or the Shortest Edit Script(SES) of two
// sequences. For more details, refer to the paper 'An O(ND) Difference
// Algorithm and Its Variations' by Eugene W. Myers.
// In the scenario of profile fuzzy matching, the two sequences are the IR
// callsite anchors and profile callsite anchors. The subsequence equivalent
// parts from the resulting SES are used to remap the IR locations to the
// profile locations. As the number of function callsite is usually not big,
// we currently just implements the basic greedy version(page 6 of the paper).
template <typename Loc, typename Function,
          typename AnchorList = ArrayRef<std::pair<Loc, Function>>>
void longestCommonSequence(
    AnchorList AnchorList1, AnchorList AnchorList2,
    llvm::function_ref<bool(const Function &, const Function &)>
        FunctionMatchesProfile,
    llvm::function_ref<void(Loc, Loc)> InsertMatching) {
  int32_t Size1 = AnchorList1.size(), Size2 = AnchorList2.size(),
          MaxDepth = Size1 + Size2;
  auto Index = [&](int32_t I) { return I + MaxDepth; };

  if (MaxDepth == 0)
    return;

  // Backtrack the SES result.
  auto Backtrack = [&](ArrayRef<std::vector<int32_t>> Trace,
                       AnchorList AnchorList1, AnchorList AnchorList2) {
    int32_t X = Size1, Y = Size2;
    for (int32_t Depth = Trace.size() - 1; X > 0 || Y > 0; Depth--) {
      const auto &P = Trace[Depth];
      int32_t K = X - Y;
      int32_t PrevK = K;
      if (K == -Depth || (K != Depth && P[Index(K - 1)] < P[Index(K + 1)]))
        PrevK = K + 1;
      else
        PrevK = K - 1;

      int32_t PrevX = P[Index(PrevK)];
      int32_t PrevY = PrevX - PrevK;
      while (X > PrevX && Y > PrevY) {
        X--;
        Y--;
        InsertMatching(AnchorList1[X].first, AnchorList2[Y].first);
      }

      if (Depth == 0)
        break;

      if (Y == PrevY)
        X--;
      else if (X == PrevX)
        Y--;
      X = PrevX;
      Y = PrevY;
    }
  };

  // The greedy LCS/SES algorithm.

  // An array contains the endpoints of the furthest reaching D-paths.
  std::vector<int32_t> V(2 * MaxDepth + 1, -1);
  V[Index(1)] = 0;
  // Trace is used to backtrack the SES result.
  std::vector<std::vector<int32_t>> Trace;
  for (int32_t Depth = 0; Depth <= MaxDepth; Depth++) {
    Trace.push_back(V);
    for (int32_t K = -Depth; K <= Depth; K += 2) {
      int32_t X = 0, Y = 0;
      if (K == -Depth || (K != Depth && V[Index(K - 1)] < V[Index(K + 1)]))
        X = V[Index(K + 1)];
      else
        X = V[Index(K - 1)] + 1;
      Y = X - K;
      while (
          X < Size1 && Y < Size2 &&
          FunctionMatchesProfile(AnchorList1[X].second, AnchorList2[Y].second))
        X++, Y++;

      V[Index(K)] = X;

      if (X >= Size1 && Y >= Size2) {
        // Length of an SES is D.
        Backtrack(Trace, AnchorList1, AnchorList2);
        return;
      }
    }
  }
  // Length of an SES is greater than MaxDepth.
}

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LONGESTCOMMONSEQEUNCE_H
