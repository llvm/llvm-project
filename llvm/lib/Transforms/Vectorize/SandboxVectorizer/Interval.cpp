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

template class Interval<Instruction>;
template class Interval<MemDGNode>;

#ifndef NDEBUG
template <typename T> void Interval<T>::dump() const { print(dbgs()); }
#endif
} // namespace llvm::sandboxir
