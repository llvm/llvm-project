//===- SLPUtils.h - SLP Vectorizer free utility helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free helper
// functions that do not depend on BoUpSLP, InstructionsState, or any other
// SLP-private type. Splitting them out keeps SLPVectorizer.cpp focused on
// the build / legality / cost / codegen pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPUTILS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace llvm {
class Instruction;
class Type;
class Value;
} // namespace llvm

namespace llvm::slpvectorizer {

/// Limit of the number of uses for potentially transformed instructions/values,
/// used in checks to avoid compile-time explode.
inline constexpr int UsesLimit = 64;

/// \returns True if the value is a constant (but not globals/constant
/// expressions).
bool isConstant(Value *V);

/// Checks if \p V is one of vector-like instructions, i.e. undef,
/// insertelement/extractelement with constant indices for fixed vector type
/// or extractvalue instruction.
bool isVectorLikeInstWithConstOps(Value *V);

/// \returns the number of elements for Ty.
unsigned getNumElements(Type *Ty);

/// Returns power-of-2 number of elements in a single register (part), given
/// the total number of elements \p Size and number of registers (parts) \p
/// NumParts.
unsigned getPartNumElems(unsigned Size, unsigned NumParts);

/// Returns correct remaining number of elements, considering total amount
/// \p Size, (power-of-2 number) of elements in a single register
/// \p PartNumElems and current register (part) \p Part.
unsigned getNumElems(unsigned Size, unsigned PartNumElems, unsigned Part);

#if !defined(NDEBUG)
/// Print a short descriptor of the instruction bundle suitable for debug
/// output.
std::string shortBundleName(ArrayRef<Value *> VL, int Idx = -1);
#endif

/// \returns True if all of the instructions in \p VL are in the same block.
bool allSameBlock(ArrayRef<Value *> VL);

/// \returns True if all of the values in \p VL are constants (but not
/// globals/constant expressions).
bool allConstant(ArrayRef<Value *> VL);

/// \returns True if all of the values in \p VL are identical or some of them
/// are UndefValue.
bool isSplat(ArrayRef<Value *> VL);

/// \returns True if \p I is commutative, handles CmpInst and BinaryOperator.
/// For BinaryOperator, it also checks if \p ValWithUses is used in specific
/// patterns that make it effectively commutative (like equality comparisons
/// with zero).
/// In most cases, users should not call this function directly (since \p I and
/// \p ValWithUses are the same). However, when analyzing interchangeable
/// instructions, we need to use the converted opcode along with the original
/// uses.
/// \param I The instruction to check for commutativity
/// \param ValWithUses The value whose uses are analyzed for special
/// patterns
bool isCommutative(const Instruction *I, const Value *ValWithUses,
                   bool IsCopyable = false);

/// This is a helper function to check whether \p I is commutative.
/// This is a convenience wrapper that calls the two-parameter version of
/// isCommutative with the same instruction for both parameters. This is
/// the common case where the instruction being checked for commutativity
/// is the same as the instruction whose uses are analyzed for special
/// patterns (see the two-parameter version above for details).
/// \param I The instruction to check for commutativity
/// \returns true if the instruction is commutative, false otherwise
bool isCommutative(const Instruction *I);

/// Checks if the operand is commutative. In commutative operations, not all
/// operands might commutable, e.g. for fmuladd only 2 first operands are
/// commutable.
bool isCommutableOperand(const Instruction *I, Value *ValWithUses, unsigned Op,
                         bool IsCopyable = false);

/// \returns number of operands of \p I, considering commutativity. Returns 2
/// for commutative intrinsics.
/// \param I The instruction to check for commutativity
unsigned getNumberOfPotentiallyCommutativeOps(Instruction *I);

/// \returns inserting or extracting index of InsertElement, ExtractElement
/// or InsertValue instruction, using \p Offset as base offset for index.
/// \returns std::nullopt if the index is not an immediate.
std::optional<unsigned> getElementIndex(const Value *Inst, unsigned Offset = 0);

/// \returns True if all of the values in \p VL use the same opcode.
/// For comparison instructions, also checks if predicates match.
/// PoisonValues are considered matching. Interchangeable instructions are
/// not considered.
bool allSameOpcode(ArrayRef<Value *> VL);

/// \returns Optional element Idx for Extract{Value,Element} instructions.
std::optional<unsigned> getExtractIndex(const Instruction *E);

/// Compute the inverse permutation \p Mask of \p Indices.
void inversePermutation(ArrayRef<unsigned> Indices, SmallVectorImpl<int> &Mask);

/// Reorders the list of scalars in accordance with the given \p Mask.
void reorderScalars(SmallVectorImpl<Value *> &Scalars, ArrayRef<int> Mask);

/// \returns True iff every value in \p VL has the same Type as the first.
bool allSameType(ArrayRef<Value *> VL);

/// Checks if the provided value does not require scheduling. It does not
/// require scheduling if this is not an instruction or it is an instruction
/// that does not read/write memory and all operands are either not
/// instructions or phi nodes or instructions from different blocks.
bool areAllOperandsNonInsts(Value *V);

/// Checks if the provided value does not require scheduling. It does not
/// require scheduling if this is not an instruction or it is an instruction
/// that does not read/write memory and all users are phi nodes or
/// instructions from different blocks.
bool isUsedOutsideBlock(Value *V);

/// Checks if the specified value does not require scheduling. It does not
/// require scheduling if all operands and all users do not need to be
/// scheduled in the current basic block.
bool doesNotNeedToBeScheduled(Value *V);

/// Checks if the specified array of instructions does not require scheduling.
/// It is so if all either instructions have operands that do not require
/// scheduling or their users do not require scheduling since they are phis or
/// in other basic blocks.
bool doesNotNeedToSchedule(ArrayRef<Value *> VL);

/// \returns inserting or extracting index of InsertElement / ExtractElement
/// instruction, using \p Offset as base offset for index. Only instantiated
/// for InsertElementInst and ExtractElementInst (see SLPUtils.cpp).
template <typename T>
std::optional<unsigned> getInsertExtractIndex(const Value *Inst,
                                              unsigned Offset);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPUTILS_H
