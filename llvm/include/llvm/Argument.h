//===-- llvm/Argument.h - Definition of the Argument class ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Argument class. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARGUMENT_H
#define LLVM_ARGUMENT_H

#include "llvm/Value.h"
#include "llvm/Attributes.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/Twine.h"

namespace llvm {

template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

/// A class to represent an incoming formal argument to a Function. An argument
/// is a very simple Value. It is essentially a named (optional) type. When used
/// in the body of a function, it represents the value of the actual argument
/// the function was called with.
/// @brief LLVM Argument representation  
class Argument : public Value, public ilist_node<Argument> {
  virtual void anchor();
  Function *Parent;

  friend class SymbolTableListTraits<Argument, Function>;
  void setParent(Function *parent);

public:
  /// Argument ctor - If Function argument is specified, this argument is
  /// inserted at the end of the argument list for the function.
  ///
  explicit Argument(Type *Ty, const Twine &Name = "", Function *F = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }

  /// getArgNo - Return the index of this formal argument in its containing
  /// function.  For example in "void foo(int a, float b)" a is 0 and b is 1. 
  unsigned getArgNo() const;
  
  /// hasByValAttr - Return true if this argument has the byval attribute on it
  /// in its containing function.
  bool hasByValAttr() const;
  
  /// getParamAlignment - If this is a byval argument, return its alignment.
  unsigned getParamAlignment() const;

  /// hasNestAttr - Return true if this argument has the nest attribute on
  /// it in its containing function.
  bool hasNestAttr() const;

  /// hasNoAliasAttr - Return true if this argument has the noalias attribute on
  /// it in its containing function.
  bool hasNoAliasAttr() const;
  
  /// hasNoCaptureAttr - Return true if this argument has the nocapture
  /// attribute on it in its containing function.
  bool hasNoCaptureAttr() const;
  
  /// hasStructRetAttr - Return true if this argument has the sret attribute on
  /// it in its containing function.
  bool hasStructRetAttr() const;

  /// addAttr - Add a Attribute to an argument
  void addAttr(Attributes);
  
  /// removeAttr - Remove a Attribute from an argument
  void removeAttr(Attributes);

  /// classof - Methods for support type inquiry through isa, cast, and
  /// dyn_cast:
  ///
  static inline bool classof(const Value *V) {
    return V->getValueID() == ArgumentVal;
  }
};

} // End llvm namespace

#endif
