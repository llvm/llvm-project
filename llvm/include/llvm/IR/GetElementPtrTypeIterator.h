//===- GetElementPtrTypeIterator.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an iterator for walking through the types indexed by
// getelementptr instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GETELEMENTPTRTYPEITERATOR_H
#define LLVM_IR_GETELEMENTPTRTYPEITERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace llvm {

template <typename ItTy = User::const_op_iterator>
class generic_gep_type_iterator {

  ItTy OpIt;
  PointerUnion<StructType *, Type *> CurTy;

  generic_gep_type_iterator() = default;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Type *;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  static generic_gep_type_iterator begin(Type *Ty, ItTy It) {
    generic_gep_type_iterator I;
    I.CurTy = Ty;
    I.OpIt = It;
    return I;
  }

  static generic_gep_type_iterator end(ItTy It) {
    generic_gep_type_iterator I;
    I.OpIt = It;
    return I;
  }

  bool operator==(const generic_gep_type_iterator &x) const {
    return OpIt == x.OpIt;
  }

  bool operator!=(const generic_gep_type_iterator &x) const {
    return !operator==(x);
  }

  // FIXME: Make this the iterator's operator*() after the 4.0 release.
  // operator*() had a different meaning in earlier releases, so we're
  // temporarily not giving this iterator an operator*() to avoid a subtle
  // semantics break.
  Type *getIndexedType() const {
    if (auto *T = CurTy.dyn_cast<Type *>())
      return T;
    return CurTy.get<StructType *>()->getTypeAtIndex(getOperand());
  }

  Value *getOperand() const { return const_cast<Value *>(&**OpIt); }

  generic_gep_type_iterator &operator++() { // Preincrement
    Type *Ty = getIndexedType();
    if (auto *ATy = dyn_cast<ArrayType>(Ty))
      CurTy = ATy->getElementType();
    else if (auto *VTy = dyn_cast<VectorType>(Ty))
      CurTy = VTy->getElementType();
    else
      CurTy = dyn_cast<StructType>(Ty);
    ++OpIt;
    return *this;
  }

  generic_gep_type_iterator operator++(int) { // Postincrement
    generic_gep_type_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  // All of the below API is for querying properties of the "outer type", i.e.
  // the type that contains the indexed type. Most of the time this is just
  // the type that was visited immediately prior to the indexed type, but for
  // the first element this is an unbounded array of the GEP's source element
  // type, for which there is no clearly corresponding IR type (we've
  // historically used a pointer type as the outer type in this case, but
  // pointers will soon lose their element type).
  //
  // FIXME: Most current users of this class are just interested in byte
  // offsets (a few need to know whether the outer type is a struct because
  // they are trying to replace a constant with a variable, which is only
  // legal for arrays, e.g. canReplaceOperandWithVariable in SimplifyCFG.cpp);
  // we should provide a more minimal API here that exposes not much more than
  // that.

  bool isStruct() const { return CurTy.is<StructType *>(); }
  bool isSequential() const { return CurTy.is<Type *>(); }

  StructType *getStructType() const { return CurTy.get<StructType *>(); }

  StructType *getStructTypeOrNull() const {
    return CurTy.dyn_cast<StructType *>();
  }
};

  using gep_type_iterator = generic_gep_type_iterator<>;

  inline gep_type_iterator gep_type_begin(const User *GEP) {
    auto *GEPOp = cast<GEPOperator>(GEP);
    return gep_type_iterator::begin(
        GEPOp->getSourceElementType(),
        GEP->op_begin() + 1);
  }

  inline gep_type_iterator gep_type_end(const User *GEP) {
    return gep_type_iterator::end(GEP->op_end());
  }

  inline gep_type_iterator gep_type_begin(const User &GEP) {
    auto &GEPOp = cast<GEPOperator>(GEP);
    return gep_type_iterator::begin(
        GEPOp.getSourceElementType(),
        GEP.op_begin() + 1);
  }

  inline gep_type_iterator gep_type_end(const User &GEP) {
    return gep_type_iterator::end(GEP.op_end());
  }

  template<typename T>
  inline generic_gep_type_iterator<const T *>
  gep_type_begin(Type *Op0, ArrayRef<T> A) {
    return generic_gep_type_iterator<const T *>::begin(Op0, A.begin());
  }

  template<typename T>
  inline generic_gep_type_iterator<const T *>
  gep_type_end(Type * /*Op0*/, ArrayRef<T> A) {
    return generic_gep_type_iterator<const T *>::end(A.end());
  }

} // end namespace llvm

#endif // LLVM_IR_GETELEMENTPTRTYPEITERATOR_H
