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

#include <optional>

namespace llvm {
class Instruction;
class Type;
class Value;
} // namespace llvm

namespace llvm::slpvectorizer {

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

/// \returns True if all of the instructions in \p VL are in the same block.
bool allSameBlock(ArrayRef<Value *> VL);

/// \returns True if all of the values in \p VL are constants (but not
/// globals/constant expressions).
bool allConstant(ArrayRef<Value *> VL);

/// \returns True if all of the values in \p VL are identical or some of them
/// are UndefValue.
bool isSplat(ArrayRef<Value *> VL);

/// \returns True if all of the values in \p VL use the same opcode.
/// For comparison instructions, also checks if predicates match.
/// PoisonValues are considered matching. Interchangeable instructions are
/// not considered.
bool allSameOpcode(ArrayRef<Value *> VL);

/// \returns Optional element Idx for Extract{Value,Element} instructions.
std::optional<unsigned> getExtractIndex(const Instruction *E);

/// \returns True iff every value in \p VL has the same Type as the first.
bool allSameType(ArrayRef<Value *> VL);

/// \returns inserting or extracting index of InsertElement / ExtractElement
/// instruction, using \p Offset as base offset for index. Only instantiated
/// for InsertElementInst and ExtractElementInst (see SLPUtils.cpp).
template <typename T>
std::optional<unsigned> getInsertExtractIndex(const Value *Inst,
                                              unsigned Offset);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPUTILS_H
