//===- SafeStackLayout.h - SafeStack frame layout --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SAFESTACKLAYOUT_H
#define LLVM_LIB_CODEGEN_SAFESTACKLAYOUT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/StackLifetime.h"

namespace llvm {

class raw_ostream;
class Value;

namespace safestack {

/// Compute the layout of an unsafe stack frame.
class StackLayout {
  uint64_t MaxAlignment;

  struct StackRegion {
    unsigned Start;
    unsigned End;
    StackLifetime::LiveRange Range;

    StackRegion(unsigned Start, unsigned End,
                const StackLifetime::LiveRange &Range)
        : Start(Start), End(End), Range(Range) {}
  };

  /// The list of current stack regions, sorted by StackRegion::Start.
  SmallVector<StackRegion, 16> Regions;

  struct StackObject {
    const Value *Handle;
    unsigned Size;
    uint64_t Alignment;
    StackLifetime::LiveRange Range;
  };

  SmallVector<StackObject, 8> StackObjects;

  DenseMap<const Value *, unsigned> ObjectOffsets;
  DenseMap<const Value *, uint64_t> ObjectAlignments;

  void layoutObject(StackObject &Obj);

public:
  StackLayout(uint64_t StackAlignment) : MaxAlignment(StackAlignment) {}

  /// Add an object to the stack frame. Value pointer is opaque and used as a
  /// handle to retrieve the object's offset in the frame later.
  void addObject(const Value *V, unsigned Size, uint64_t Alignment,
                 const StackLifetime::LiveRange &Range);

  /// Run the layout computation for all previously added objects.
  void computeLayout();

  /// Returns the offset to the object start in the stack frame.
  unsigned getObjectOffset(const Value *V) { return ObjectOffsets[V]; }

  /// Returns the alignment of the object
  uint64_t getObjectAlignment(const Value *V) { return ObjectAlignments[V]; }

  /// Returns the size of the entire frame.
  unsigned getFrameSize() { return Regions.empty() ? 0 : Regions.back().End; }

  /// Returns the alignment of the frame.
  uint64_t getFrameAlignment() { return MaxAlignment; }

  void print(raw_ostream &OS);
};

} // end namespace safestack

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_SAFESTACKLAYOUT_H
