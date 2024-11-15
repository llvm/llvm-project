//===- InstrInterval.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The InstrInterval class is an interval of instructions in a block.
// It provides an API for some basic operations on the interval, including some
// simple set operations, like union, interseciton and others.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H

#include "llvm/SandboxIR/SandboxIR.h"
#include <iterator>

namespace llvm::sandboxir {

/// A simple iterator for iterating the interval.
template <typename DerefType, typename InstrIntervalType>
class InstrIntervalIterator {
  sandboxir::Instruction *I;
  InstrIntervalType &R;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Instruction;
  using pointer = value_type *;
  using reference = sandboxir::Instruction &;
  using iterator_category = std::bidirectional_iterator_tag;

  InstrIntervalIterator(sandboxir::Instruction *I, InstrIntervalType &R)
      : I(I), R(R) {}
  bool operator==(const InstrIntervalIterator &Other) const {
    assert(&R == &Other.R && "Iterators belong to different regions!");
    return Other.I == I;
  }
  bool operator!=(const InstrIntervalIterator &Other) const {
    return !(*this == Other);
  }
  InstrIntervalIterator &operator++() {
    assert(I != nullptr && "already at end()!");
    I = I->getNextNode();
    return *this;
  }
  InstrIntervalIterator operator++(int) {
    auto ItCopy = *this;
    ++*this;
    return ItCopy;
  }
  InstrIntervalIterator &operator--() {
    // `I` is nullptr for end() when ToI is the BB terminator.
    I = I != nullptr ? I->getPrevNode() : R.ToI;
    return *this;
  }
  InstrIntervalIterator operator--(int) {
    auto ItCopy = *this;
    --*this;
    return ItCopy;
  }
  template <typename T =
                std::enable_if<std::is_same<DerefType, Instruction *&>::value>>
  sandboxir::Instruction &operator*() {
    return *I;
  }
  DerefType operator*() const { return *I; }
};

class InstrInterval {
  Instruction *FromI;
  Instruction *ToI;

public:
  InstrInterval() : FromI(nullptr), ToI(nullptr) {}
  InstrInterval(Instruction *FromI, Instruction *ToI) : FromI(FromI), ToI(ToI) {
    assert((FromI == ToI || FromI->comesBefore(ToI)) &&
           "FromI should come before TopI!");
  }
  InstrInterval(ArrayRef<Instruction *> Instrs) {
    assert(!Instrs.empty() && "Expected non-empty Instrs!");
    FromI = Instrs[0];
    ToI = Instrs[0];
    for (auto *I : drop_begin(Instrs)) {
      if (I->comesBefore(FromI))
        FromI = I;
      else if (ToI->comesBefore(I))
        ToI = I;
    }
  }
  bool empty() const {
    assert(((FromI == nullptr && ToI == nullptr) ||
            (FromI != nullptr && ToI != nullptr)) &&
           "Either none or both should be null");
    return FromI == nullptr;
  }
  bool contains(Instruction *I) const {
    if (empty())
      return false;
    return (FromI == I || FromI->comesBefore(I)) &&
           (I == ToI || I->comesBefore(ToI));
  }
  Instruction *top() const { return FromI; }
  Instruction *bottom() const { return ToI; }

  using iterator =
      InstrIntervalIterator<sandboxir::Instruction &, InstrInterval>;
  using const_iterator = InstrIntervalIterator<const sandboxir::Instruction &,
                                               const InstrInterval>;
  iterator begin() { return iterator(FromI, *this); }
  iterator end() {
    return iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
  }
  const_iterator begin() const { return const_iterator(FromI, *this); }
  const_iterator end() const {
    return const_iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
  }
};
} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_INSTRINTERVAL_H
