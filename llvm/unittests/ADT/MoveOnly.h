//===- llvm/unittest/ADT/MoveOnly.h - Optional unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_ADT_MOVEONLY_H
#define LLVM_UNITTESTS_ADT_MOVEONLY_H

namespace llvm {

struct MoveOnly {
  static unsigned MoveConstructions;
  static unsigned Destructions;
  static unsigned MoveAssignments;
  int val;
  explicit MoveOnly(int val) : val(val) {
  }
  MoveOnly(MoveOnly&& other) {
    val = other.val;
    ++MoveConstructions;
  }
  MoveOnly &operator=(MoveOnly&& other) {
    val = other.val;
    ++MoveAssignments;
    return *this;
  }
  ~MoveOnly() {
    ++Destructions;
  }
  static void ResetCounts() {
    MoveConstructions = 0;
    Destructions = 0;
    MoveAssignments = 0;
  }
};

}  // end namespace llvm

#endif // LLVM_UNITTESTS_ADT_MOVEONLY_H
