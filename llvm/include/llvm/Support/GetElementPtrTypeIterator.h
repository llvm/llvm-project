//===- llvm/Support/GetElementPtrTypeIterator.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an iterator for walking through the types indexed by
// getelementptr instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GETELEMENTPTRTYPE_H
#define LLVM_SUPPORT_GETELEMENTPTRTYPE_H

#include "llvm/User.h"
#include "llvm/DerivedTypes.h"

namespace llvm {
  template<typename ItTy = User::const_op_iterator>
  class generic_gep_type_iterator
    : public std::iterator<std::forward_iterator_tag, Type *, ptrdiff_t> {
    typedef std::iterator<std::forward_iterator_tag,
                          Type *, ptrdiff_t> super;

    ItTy OpIt;
    Type *CurTy;
    generic_gep_type_iterator() {}
  public:

    static generic_gep_type_iterator begin(Type *Ty, ItTy It) {
      generic_gep_type_iterator I;
      I.CurTy = Ty;
      I.OpIt = It;
      return I;
    }
    static generic_gep_type_iterator end(ItTy It) {
      generic_gep_type_iterator I;
      I.CurTy = 0;
      I.OpIt = It;
      return I;
    }

    bool operator==(const generic_gep_type_iterator& x) const {
      return OpIt == x.OpIt;
    }
    bool operator!=(const generic_gep_type_iterator& x) const {
      return !operator==(x);
    }

    Type *operator*() const {
      return CurTy;
    }

    Type *getIndexedType() const {
      CompositeType *CT = cast<CompositeType>(CurTy);
      return CT->getTypeAtIndex(getOperand());
    }

    // This is a non-standard operator->.  It allows you to call methods on the
    // current type directly.
    Type *operator->() const { return operator*(); }

    Value *getOperand() const { return *OpIt; }

    generic_gep_type_iterator& operator++() {   // Preincrement
      if (CompositeType *CT = dyn_cast<CompositeType>(CurTy)) {
        CurTy = CT->getTypeAtIndex(getOperand());
      } else {
        CurTy = 0;
      }
      ++OpIt;
      return *this;
    }

    generic_gep_type_iterator operator++(int) { // Postincrement
      generic_gep_type_iterator tmp = *this; ++*this; return tmp;
    }
  };

  typedef generic_gep_type_iterator<> gep_type_iterator;

  inline gep_type_iterator gep_type_begin(const User *GEP) {
    return gep_type_iterator::begin(GEP->getOperand(0)->getType(),
                                    GEP->op_begin()+1);
  }
  inline gep_type_iterator gep_type_end(const User *GEP) {
    return gep_type_iterator::end(GEP->op_end());
  }
  inline gep_type_iterator gep_type_begin(const User &GEP) {
    return gep_type_iterator::begin(GEP.getOperand(0)->getType(),
                                    GEP.op_begin()+1);
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
  gep_type_end(Type *Op0, ArrayRef<T> A) {
    return generic_gep_type_iterator<const T *>::end(A.end());
  }
} // end namespace llvm

#endif
