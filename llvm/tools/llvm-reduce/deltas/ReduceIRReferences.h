//===- ReduceIRReferences.h  - Specialized Delta Pass -----------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting IR references from the MachineFunction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEIRREFERENCES_MIR_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAS_REDUCEIRREFERENCES_MIR_H

#include "Delta.h"

namespace llvm {

/// Remove IR references from instructions (i.e. from memory operands)
void reduceIRInstructionReferencesDeltaPass(Oracle &O,
                                            ReducerWorkItem &WorkItem);

/// Remove IR BasicBlock references (the block names)
void reduceIRBlockReferencesDeltaPass(Oracle &O, ReducerWorkItem &WorkItem);

/// Remove IR references from function level fields (e.g. frame object names)
void reduceIRFunctionReferencesDeltaPass(Oracle &O, ReducerWorkItem &WorkItem);

} // namespace llvm

#endif
