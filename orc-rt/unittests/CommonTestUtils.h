//===- CommonTestUtils.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_COMMONTESTUTILS_H
#define ORC_RT_UNITTEST_COMMONTESTUTILS_H

#include <cstddef>

template <size_t Idx = 0> class OpCounter {
public:
  OpCounter() { ++DefaultConstructions; }
  OpCounter(const OpCounter &Other) { ++CopyConstructions; }
  OpCounter &operator=(const OpCounter &Other) {
    ++CopyAssignments;
    return *this;
  }
  OpCounter(OpCounter &&Other) { ++MoveConstructions; }
  OpCounter &operator=(OpCounter &&Other) {
    ++MoveAssignments;
    return *this;
  }
  ~OpCounter() { ++Destructions; }

  static size_t defaultConstructions() { return DefaultConstructions; }
  static size_t copyConstructions() { return CopyConstructions; }
  static size_t copyAssignments() { return CopyAssignments; }
  static size_t copies() { return copyConstructions() + copyAssignments(); }
  static size_t moveConstructions() { return MoveConstructions; }
  static size_t moveAssignments() { return MoveAssignments; }
  static size_t moves() { return moveConstructions() + moveAssignments(); }
  static size_t destructions() { return Destructions; }

  static bool destructionsMatch() {
    return destructions() == defaultConstructions() + copies() + moves();
  }

  static void reset() {
    DefaultConstructions = 0;
    CopyConstructions = 0;
    CopyAssignments = 0;
    MoveConstructions = 0;
    MoveAssignments = 0;
    Destructions = 0;
  }

private:
  static size_t DefaultConstructions;
  static size_t CopyConstructions;
  static size_t CopyAssignments;
  static size_t MoveConstructions;
  static size_t MoveAssignments;
  static size_t Destructions;
};

template <size_t Idx> size_t OpCounter<Idx>::DefaultConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::CopyConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::CopyAssignments = 0;
template <size_t Idx> size_t OpCounter<Idx>::MoveConstructions = 0;
template <size_t Idx> size_t OpCounter<Idx>::MoveAssignments = 0;
template <size_t Idx> size_t OpCounter<Idx>::Destructions = 0;

#endif // ORC_RT_UNITTEST_COMMONTESTUTILS_H
