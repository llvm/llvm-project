//===- ReduceMemoryOperations.h - Specialized Delta Pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEMEMORYOPERATIONS_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEMEMORYOPERATIONS_H

#include "Delta.h"

namespace llvm {
void reduceVolatileInstructionsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem);
void reduceAtomicSyncScopesDeltaPass(Oracle &O, ReducerWorkItem &WorkItem);
void reduceAtomicOrderingDeltaPass(Oracle &O, ReducerWorkItem &WorkItem);
} // namespace llvm

#endif
