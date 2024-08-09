//===- llvm/unittest/ADT/CountCopyAndMove.h - Optional unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_ADT_COUNTCOPYANDMOVE_H
#define LLVM_UNITTESTS_ADT_COUNTCOPYANDMOVE_H

namespace llvm {

struct CountCopyAndMove {
  static int Constructions;
  static int CopyConstructions;
  static int CopyAssignments;
  static int MoveConstructions;
  static int MoveAssignments;
  static int Destructions;
  int val;

  CountCopyAndMove() { ++Constructions; }
  explicit CountCopyAndMove(int val) : val(val) { ++Constructions; }
  CountCopyAndMove(const CountCopyAndMove &other) : val(other.val) {
    ++CopyConstructions;
  }
  CountCopyAndMove &operator=(const CountCopyAndMove &other) {
    val = other.val;
    ++CopyAssignments;
    return *this;
  }
  CountCopyAndMove(CountCopyAndMove &&other) : val(other.val) {
    ++MoveConstructions;
  }
  CountCopyAndMove &operator=(CountCopyAndMove &&other) {
    val = other.val;
    ++MoveAssignments;
    return *this;
  }
  ~CountCopyAndMove() { ++Destructions; }

  static void ResetCounts() {
    Constructions = 0;
    CopyConstructions = 0;
    CopyAssignments = 0;
    MoveConstructions = 0;
    MoveAssignments = 0;
    Destructions = 0;
  }

  static int TotalConstructions() {
    return Constructions + MoveConstructions + CopyConstructions;
  }

  static int TotalCopies() { return CopyConstructions + CopyAssignments; }

  static int TotalMoves() { return MoveConstructions + MoveAssignments; }
};

} // end namespace llvm

#endif // LLVM_UNITTESTS_ADT_COUNTCOPYANDMOVE_H
