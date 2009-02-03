//===-------- llvm/GlobalAlias.h - GlobalAlias class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalAlias class, which
// represents a single function or variable alias in the IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_ALIAS_H
#define LLVM_GLOBAL_ALIAS_H

#include "llvm/GlobalValue.h"
#include "llvm/OperandTraits.h"
#include "llvm/ADT/ilist_node.h"

namespace llvm {

class Module;
class Constant;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalAlias : public GlobalValue, public ilist_node<GlobalAlias> {
  friend class SymbolTableListTraits<GlobalAlias, Module>;
  void operator=(const GlobalAlias &);     // Do not implement
  GlobalAlias(const GlobalAlias &);     // Do not implement

  void setParent(Module *parent);

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  /// GlobalAlias ctor - If a parent module is specified, the alias is
  /// automatically inserted into the end of the specified module's alias list.
  GlobalAlias(const Type *Ty, LinkageTypes Linkage, const std::string &Name = "",
              Constant* Aliasee = 0, Module *Parent = 0);

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// isDeclaration - Is this global variable lacking an initializer?  If so, 
  /// the global variable is defined in some other translation unit, and is thus
  /// only a declaration here.
  virtual bool isDeclaration() const;

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  virtual void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  virtual void eraseFromParent();

  /// set/getAliasee - These methods retrive and set alias target.
  void setAliasee(Constant* GV);
  const Constant* getAliasee() const {
    return cast_or_null<Constant>(getOperand(0));
  }
  Constant* getAliasee() {
    return cast_or_null<Constant>(getOperand(0));
  }
  /// getAliasedGlobal() - Aliasee can be either global or bitcast of
  /// global. This method retrives the global for both aliasee flavours.
  const GlobalValue* getAliasedGlobal() const;

  /// resolveAliasedGlobal() - This method tries to ultimately resolve the alias
  /// by going through the aliasing chain and trying to find the very last
  /// global. Returns NULL if a cycle was found. If stopOnWeak is false, then
  /// the whole chain aliasing chain is traversed, otherwise - only strong
  /// aliases.
  const GlobalValue* resolveAliasedGlobal(bool stopOnWeak = true) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalAlias *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalAliasVal;
  }
};

template <>
struct OperandTraits<GlobalAlias> : FixedNumOperandTraits<1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GlobalAlias, Value)

} // End llvm namespace

#endif
