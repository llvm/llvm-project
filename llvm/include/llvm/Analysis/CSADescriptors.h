//===- llvm/Analysis/CSADescriptors.h - CSA Descriptors --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file "describes" conditional scalar assignments (CSA).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

#ifndef LLVM_ANALYSIS_CSADESCRIPTORS_H
#define LLVM_ANALYSIS_CSADESCRIPTORS_H

namespace llvm {

/// A Conditional Scalar Assignment (CSA) is an assignment from an initial
/// scalar that may or may not occur.
class CSADescriptor {
  /// If the conditional assignment occurs inside a loop, then Phi chooses
  /// the value of the assignment from the entry block or the loop body block.
  PHINode *Phi = nullptr;

  /// The initial value of the CSA. If the condition guarding the assignment is
  /// not met, then the assignment retains this value.
  Value *InitScalar = nullptr;

  /// The Instruction that conditionally assigned to inside the loop.
  Instruction *Assignment = nullptr;

  /// Create a CSA Descriptor that models an invalid CSA.
  CSADescriptor() = default;

  /// Create a CSA Descriptor that models a valid CSA with its members
  /// initialized correctly.
  CSADescriptor(PHINode *Phi, Instruction *Assignment, Value *InitScalar)
      : Phi(Phi), InitScalar(InitScalar), Assignment(Assignment) {}

public:
  /// If Phi is the root of a CSA, return the CSADescriptor of the CSA rooted by
  /// Phi. Otherwise, return a CSADescriptor with IsValidCSA set to false.
  static CSADescriptor isCSAPhi(PHINode *Phi, Loop *TheLoop);

  operator bool() const { return isValid(); }

  /// Returns whether SI is the Assignment in CSA
  static bool isCSASelect(CSADescriptor Desc, SelectInst *SI) {
    return Desc.getAssignment() == SI;
  }

  /// Return whether this CSADescriptor models a valid CSA.
  bool isValid() const { return Phi && InitScalar && Assignment; }

  /// Return the PHI that roots this CSA.
  PHINode *getPhi() const { return Phi; }

  /// Return the initial value of the CSA. This is the value if the conditional
  /// assignment does not occur.
  Value *getInitScalar() const { return InitScalar; }

  /// The Instruction that is used after the loop
  Instruction *getAssignment() const { return Assignment; }

  /// Return the condition that this CSA is conditional upon.
  Value *getCond() const {
    if (auto *SI = dyn_cast_or_null<SelectInst>(Assignment))
      return SI->getCondition();
    return nullptr;
  }
};
} // namespace llvm

#endif // LLVM_ANALYSIS_CSADESCRIPTORS_H
