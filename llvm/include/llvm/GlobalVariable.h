//===-- llvm/GlobalVariable.h - GlobalVariable class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalVariable class, which
// represents a single global variable (or constant) in the VM.
//
// Global variables are constant pointers that refer to hunks of space that are
// allocated by either the VM, or by the linker in a static compiler.  A global
// variable may have an initial value, which is copied into the executables .data
// area.  Global Constants are required to have initializers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_VARIABLE_H
#define LLVM_GLOBAL_VARIABLE_H

#include "llvm/GlobalValue.h"
#include "llvm/OperandTraits.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/Twine.h"

namespace llvm {

class Module;
class Constant;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalVariable : public GlobalValue, public ilist_node<GlobalVariable> {
  friend class SymbolTableListTraits<GlobalVariable, Module>;
  void *operator new(size_t, unsigned) LLVM_DELETED_FUNCTION;
  void operator=(const GlobalVariable &) LLVM_DELETED_FUNCTION;
  GlobalVariable(const GlobalVariable &) LLVM_DELETED_FUNCTION;

  void setParent(Module *parent);

  bool isConstantGlobal : 1;           // Is this a global constant?
  unsigned threadLocalMode : 3;        // Is this symbol "Thread Local",
                                       // if so, what is the desired model?

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }

  enum ThreadLocalMode {
    NotThreadLocal = 0,
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel
  };

  /// GlobalVariable ctor - If a parent module is specified, the global is
  /// automatically inserted into the end of the specified modules global list.
  GlobalVariable(Type *Ty, bool isConstant, LinkageTypes Linkage,
                 Constant *Initializer = 0, const Twine &Name = "",
                 ThreadLocalMode = NotThreadLocal, unsigned AddressSpace = 0);
  /// GlobalVariable ctor - This creates a global and inserts it before the
  /// specified other global.
  GlobalVariable(Module &M, Type *Ty, bool isConstant,
                 LinkageTypes Linkage, Constant *Initializer,
                 const Twine &Name = "",
                 GlobalVariable *InsertBefore = 0,
                 ThreadLocalMode = NotThreadLocal,
                 unsigned AddressSpace = 0);

  ~GlobalVariable() {
    NumOperands = 1; // FIXME: needed by operator delete
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// hasInitializer - Unless a global variable isExternal(), it has an
  /// initializer.  The initializer for the global variable/constant is held by
  /// Initializer if an initializer is specified.
  ///
  inline bool hasInitializer() const { return !isDeclaration(); }

  /// hasDefinitiveInitializer - Whether the global variable has an initializer,
  /// and any other instances of the global (this can happen due to weak
  /// linkage) are guaranteed to have the same initializer.
  ///
  /// Note that if you want to transform a global, you must use
  /// hasUniqueInitializer() instead, because of the *_odr linkage type.
  ///
  /// Example:
  ///
  /// @a = global SomeType* null - Initializer is both definitive and unique.
  ///
  /// @b = global weak SomeType* null - Initializer is neither definitive nor
  /// unique.
  ///
  /// @c = global weak_odr SomeType* null - Initializer is definitive, but not
  /// unique.
  inline bool hasDefinitiveInitializer() const {
    return hasInitializer() &&
      // The initializer of a global variable with weak linkage may change at
      // link time.
      !mayBeOverridden();
  }

  /// hasUniqueInitializer - Whether the global variable has an initializer, and
  /// any changes made to the initializer will turn up in the final executable.
  inline bool hasUniqueInitializer() const {
    return hasInitializer() &&
      // It's not safe to modify initializers of global variables with weak
      // linkage, because the linker might choose to discard the initializer and
      // use the initializer from another instance of the global variable
      // instead. It is wrong to modify the initializer of a global variable
      // with *_odr linkage because then different instances of the global may
      // have different initializers, breaking the One Definition Rule.
      !isWeakForLinker();
  }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  inline const Constant *getInitializer() const {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return static_cast<Constant*>(Op<0>().get());
  }
  inline Constant *getInitializer() {
    assert(hasInitializer() && "GV doesn't have initializer!");
    return static_cast<Constant*>(Op<0>().get());
  }
  /// setInitializer - Sets the initializer for this global variable, removing
  /// any existing initializer if InitVal==NULL.  If this GV has type T*, the
  /// initializer must have type T.
  void setInitializer(Constant *InitVal);

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const { return isConstantGlobal; }
  void setConstant(bool Val) { isConstantGlobal = Val; }

  /// If the value is "Thread Local", its value isn't shared by the threads.
  bool isThreadLocal() const { return threadLocalMode != NotThreadLocal; }
  void setThreadLocal(bool Val) {
    threadLocalMode = Val ? GeneralDynamicTLSModel : NotThreadLocal;
  }
  void setThreadLocalMode(ThreadLocalMode Val) { threadLocalMode = Val; }
  ThreadLocalMode getThreadLocalMode() const {
    return static_cast<ThreadLocalMode>(threadLocalMode);
  }

  /// copyAttributesFrom - copy all additional attributes (those not needed to
  /// create a GlobalVariable) from the GlobalVariable Src to this one.
  void copyAttributesFrom(const GlobalValue *Src);

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  virtual void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  virtual void eraseFromParent();

  /// Override Constant's implementation of this method so we can
  /// replace constant initializers.
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalVariableVal;
  }
};

template <>
struct OperandTraits<GlobalVariable> :
  public OptionalOperandTraits<GlobalVariable> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GlobalVariable, Value)

} // End llvm namespace

#endif
