//===- Interval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Interval.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"

namespace llvm::sandboxir {

template <typename T> bool Interval<T>::disjoint(const Interval &Other) const {
  if (Other.empty())
    return true;
  if (empty())
    return true;
  return Other.Bottom->comesBefore(Top) || Bottom->comesBefore(Other.Top);
}

#ifndef NDEBUG
template <typename T> void Interval<T>::print(raw_ostream &OS) const {
  auto *Top = top();
  auto *Bot = bottom();
  OS << "Top: ";
  if (Top != nullptr)
    OS << *Top;
  else
    OS << "nullptr";
  OS << "\n";

  OS << "Bot: ";
  if (Bot != nullptr)
    OS << *Bot;
  else
    OS << "nullptr";
  OS << "\n";
}
template <typename T> void Interval<T>::dump() const { print(dbgs()); }
#endif

template class Interval<Instruction>;
template class Interval<MemDGNode>;

} // namespace llvm::sandboxir
