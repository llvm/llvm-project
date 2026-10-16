//===-- Operations.h - ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of common fuzzer operation descriptors for building an IR
// mutator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZMUTATE_OPERATIONS_H
#define LLVM_FUZZMUTATE_OPERATIONS_H

#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// Getters for the default sets of operations, per general category.
/// @{
LLVM_ABI void describeFuzzerIntOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void describeFuzzerFloatOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void
describeFuzzerControlFlowOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void
describeFuzzerPointerOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void
describeFuzzerAggregateOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void describeFuzzerVectorOps(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void
describeFuzzerUnaryOperations(std::vector<fuzzerop::OpDescriptor> &Ops);
LLVM_ABI void describeFuzzerOtherOps(std::vector<fuzzerop::OpDescriptor> &Ops);
/// @}

namespace fuzzerop {

/// Descriptors for individual operations.
/// @{
LLVM_ABI OpDescriptor selectDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor fnegDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor binOpDescriptor(unsigned Weight,
                                      Instruction::BinaryOps Op);
LLVM_ABI OpDescriptor cmpOpDescriptor(unsigned Weight,
                                      Instruction::OtherOps CmpOp,
                                      CmpInst::Predicate Pred);
LLVM_ABI OpDescriptor splitBlockDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor gepDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor extractValueDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor insertValueDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor extractElementDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor insertElementDescriptor(unsigned Weight);
LLVM_ABI OpDescriptor shuffleVectorDescriptor(unsigned Weight);

/// @}

} // namespace fuzzerop

} // namespace llvm

#endif // LLVM_FUZZMUTATE_OPERATIONS_H
