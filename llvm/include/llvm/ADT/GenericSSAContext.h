//===- GenericSSAContext.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the little GenericSSAContext<X> template class
/// that can be used to implement IR analyses as templates.
/// Specializing these templates allows the analyses to be used over
/// both LLVM IR and Machine IR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GENERICSSACONTEXT_H
#define LLVM_ADT_GENERICSSACONTEXT_H

#include "llvm/Support/Printable.h"

namespace llvm {

template <typename _FunctionT> class GenericSSAContext {
public:
  // Specializations should provide the following types that are similar to how
  // LLVM IR is structured:

  // The smallest unit of the IR is a ValueT. The SSA context uses a ValueRefT,
  // which is a pointer to a ValueT, since Machine IR does not have the
  // equivalent of a ValueT.
  //
  // using ValueRefT = ...
  //
  // The ConstValueRefT is needed to work with "const Value *", where const
  // needs to bind to the pointee and not the pointer.
  //
  // using ConstValueRefT = ...
  //
  // The null value for ValueRefT.
  //
  // static constexpr ValueRefT ValueRefNull;

  // An InstructionT usually defines one or more ValueT objects.
  //
  // using InstructionT = ... must be a subclass of Value

  // A UseT represents a data-edge from the defining instruction to the using
  // instruction.
  //
  // using UseT = ...

  // A BlockT is a sequence of InstructionT, and forms a node of the CFG. It
  // has global methods predecessors() and successors() that return
  // the list of incoming CFG edges and outgoing CFG edges
  // respectively.
  //
  // using BlockT = ...

  // A FunctionT represents a CFG along with arguments and return values. It is
  // the smallest complete unit of code in a Module.
  //
  // The compiler produces an error here if this class is implicitly
  // specialized due to an instantiation. An explicit specialization
  // of this template needs to be added before the instantiation point
  // indicated by the compiler.
  using FunctionT = typename _FunctionT::invalidTemplateInstanceError;

  // A dominator tree provides the dominance relation between basic blocks in
  // a given funciton.
  //
  // using DominatorTreeT = ...

  // Initialize the SSA context with information about the FunctionT being
  // processed.
  //
  // void setFunction(FunctionT &function);
  // FunctionT* getFunction() const;

  // Every FunctionT has a unique BlockT marked as its entry.
  //
  // static BlockT* getEntryBlock(FunctionT &F);

  // Methods to examine basic blocks and values
  //
  // static void appendBlockDefs(SmallVectorImpl<ValueRefT> &defs,
  //                             BlockT &block);
  // static void appendBlockDefs(SmallVectorImpl<const ValueRefT> &defs,
  //                             const BlockT &block);

  // static void appendBlockTerms(SmallVectorImpl<InstructionT *> &terms,
  //                              BlockT &block);
  // static void appendBlockTerms(SmallVectorImpl<const InstructionT *> &terms,
  //                              const BlockT &block);
  //
  // static bool comesBefore(const InstructionT *lhs, const InstructionT *rhs);
  // static bool isConstantOrUndefValuePhi(const InstructionT &Instr);
  // const BlockT *getDefBlock(const ValueRefT value) const;

  // Methods to print various objects.
  //
  // Printable print(BlockT *block) const;
  // Printable print(InstructionT *inst) const;
  // Printable print(ValueRefT value) const;
};
} // namespace llvm

#endif // LLVM_ADT_GENERICSSACONTEXT_H
