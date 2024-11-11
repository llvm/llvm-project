//===- Value.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_VALUE_H
#define LLVM_SANDBOXIR_VALUE_H

#include "llvm/IR/Value.h"
#include "llvm/SandboxIR/Use.h"

namespace llvm::sandboxir {

// Forward declare all classes to avoid some MSVC build errors.
#define DEF_INSTR(ID, OPC, CLASS) class CLASS;
#define DEF_CONST(ID, CLASS) class CLASS;
#define DEF_USER(ID, CLASS) class CLASS;
#include "llvm/SandboxIR/Values.def"
class Context;
class FuncletPadInst;
class Type;
class GlobalValue;
class GlobalObject;
class Module;
class UnaryInstruction;
class CmpInst;
class IntrinsicInst;
class Operator;
class OverflowingBinaryOperator;
class FPMathOperator;

/// Iterator for the `Use` edges of a Value's users.
/// \Returns a `Use` when dereferenced.
class UserUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty UserUseIterator.
  UserUseIterator(const class Use &Use) : Use(Use) {}
  friend class Value; // For constructor

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  UserUseIterator() = default;
  value_type operator*() const { return Use; }
  UserUseIterator &operator++();
  bool operator==(const UserUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const UserUseIterator &Other) const {
    return !(*this == Other);
  }
  const sandboxir::Use &getUse() const { return Use; }
};

/// A SandboxIR Value has users. This is the base class.
class Value {
public:
  enum class ClassID : unsigned {
#define DEF_VALUE(ID, CLASS) ID,
#define DEF_USER(ID, CLASS) ID,
#define DEF_CONST(ID, CLASS) ID,
#define DEF_INSTR(ID, OPC, CLASS) ID,
#include "llvm/SandboxIR/Values.def"
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
#define DEF_VALUE(ID, CLASS)                                                   \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_CONST(ID, CLASS)                                                   \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return #ID;
#include "llvm/SandboxIR/Values.def"
    }
    llvm_unreachable("Unimplemented ID");
  }

  /// For isa/dyn_cast.
  ClassID SubclassID;
#ifndef NDEBUG
  /// A unique ID used for forming the name (used for debugging).
  unsigned UID;
#endif
  /// The LLVM Value that corresponds to this SandboxIR Value.
  /// NOTE: Some sandboxir Instructions, like Packs, may include more than one
  /// value and in these cases `Val` points to the last instruction in program
  /// order.
  llvm::Value *Val = nullptr;

  friend class Context;               // For getting `Val`.
  friend class User;                  // For getting `Val`.
  friend class Use;                   // For getting `Val`.
  friend class VAArgInst;             // For getting `Val`.
  friend class FreezeInst;            // For getting `Val`.
  friend class FenceInst;             // For getting `Val`.
  friend class SelectInst;            // For getting `Val`.
  friend class ExtractElementInst;    // For getting `Val`.
  friend class InsertElementInst;     // For getting `Val`.
  friend class ShuffleVectorInst;     // For getting `Val`.
  friend class ExtractValueInst;      // For getting `Val`.
  friend class InsertValueInst;       // For getting `Val`.
  friend class BranchInst;            // For getting `Val`.
  friend class LoadInst;              // For getting `Val`.
  friend class StoreInst;             // For getting `Val`.
  friend class ReturnInst;            // For getting `Val`.
  friend class CallBase;              // For getting `Val`.
  friend class CallInst;              // For getting `Val`.
  friend class InvokeInst;            // For getting `Val`.
  friend class CallBrInst;            // For getting `Val`.
  friend class LandingPadInst;        // For getting `Val`.
  friend class FuncletPadInst;        // For getting `Val`.
  friend class CatchPadInst;          // For getting `Val`.
  friend class CleanupPadInst;        // For getting `Val`.
  friend class CatchReturnInst;       // For getting `Val`.
  friend class GetElementPtrInst;     // For getting `Val`.
  friend class ResumeInst;            // For getting `Val`.
  friend class CatchSwitchInst;       // For getting `Val`.
  friend class CleanupReturnInst;     // For getting `Val`.
  friend class SwitchInst;            // For getting `Val`.
  friend class UnaryOperator;         // For getting `Val`.
  friend class BinaryOperator;        // For getting `Val`.
  friend class AtomicRMWInst;         // For getting `Val`.
  friend class AtomicCmpXchgInst;     // For getting `Val`.
  friend class AllocaInst;            // For getting `Val`.
  friend class CastInst;              // For getting `Val`.
  friend class PHINode;               // For getting `Val`.
  friend class UnreachableInst;       // For getting `Val`.
  friend class CatchSwitchAddHandler; // For `Val`.
  friend class CmpInst;               // For getting `Val`.
  friend class ConstantArray;         // For `Val`.
  friend class ConstantStruct;        // For `Val`.
  friend class ConstantAggregateZero; // For `Val`.
  friend class ConstantPointerNull;   // For `Val`.
  friend class UndefValue;            // For `Val`.
  friend class PoisonValue;           // For `Val`.
  friend class BlockAddress;          // For `Val`.
  friend class GlobalValue;           // For `Val`.
  friend class DSOLocalEquivalent;    // For `Val`.
  friend class GlobalObject;          // For `Val`.
  friend class GlobalIFunc;           // For `Val`.
  friend class GlobalVariable;        // For `Val`.
  friend class GlobalAlias;           // For `Val`.
  friend class NoCFIValue;            // For `Val`.
  friend class ConstantPtrAuth;       // For `Val`.
  friend class ConstantExpr;          // For `Val`.
  friend class Utils;                 // For `Val`.
  friend class Module;                // For `Val`.
  friend class IntrinsicInst;         // For `Val`.
  friend class Operator;              // For `Val`.
  friend class OverflowingBinaryOperator; // For `Val`.
  friend class FPMathOperator;            // For `Val`.
  // Region needs to manipulate metadata in the underlying LLVM Value, we don't
  // expose metadata in sandboxir.
  friend class Region;

  /// All values point to the context.
  Context &Ctx;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

  Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx);
  /// Disable copies.
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;

public:
  virtual ~Value() = default;
  ClassID getSubclassID() const { return SubclassID; }

  using use_iterator = UserUseIterator;
  using const_use_iterator = UserUseIterator;

  use_iterator use_begin();
  const_use_iterator use_begin() const {
    return const_cast<Value *>(this)->use_begin();
  }
  use_iterator use_end() { return use_iterator(Use(nullptr, nullptr, Ctx)); }
  const_use_iterator use_end() const {
    return const_cast<Value *>(this)->use_end();
  }

  iterator_range<use_iterator> uses() {
    return make_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return make_range<const_use_iterator>(use_begin(), use_end());
  }

  /// Helper for mapped_iterator.
  struct UseToUser {
    User *operator()(const Use &Use) const { return &*Use.getUser(); }
  };

  using user_iterator = mapped_iterator<sandboxir::UserUseIterator, UseToUser>;
  using const_user_iterator = user_iterator;

  user_iterator user_begin();
  user_iterator user_end() {
    return user_iterator(Use(nullptr, nullptr, Ctx), UseToUser());
  }
  const_user_iterator user_begin() const {
    return const_cast<Value *>(this)->user_begin();
  }
  const_user_iterator user_end() const {
    return const_cast<Value *>(this)->user_end();
  }

  iterator_range<user_iterator> users() {
    return make_range<user_iterator>(user_begin(), user_end());
  }
  iterator_range<const_user_iterator> users() const {
    return make_range<const_user_iterator>(user_begin(), user_end());
  }
  /// \Returns the number of user edges (not necessarily to unique users).
  /// WARNING: This is a linear-time operation.
  unsigned getNumUses() const;
  /// Return true if this value has N uses or more.
  /// This is logically equivalent to getNumUses() >= N.
  /// WARNING: This can be expensive, as it is linear to the number of users.
  bool hasNUsesOrMore(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt >= Num)
        return true;
    }
    return false;
  }
  /// Return true if this Value has exactly N uses.
  bool hasNUses(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt > Num)
        return false;
    }
    return Cnt == Num;
  }

  Type *getType() const;

  Context &getContext() const { return Ctx; }

  void replaceUsesWithIf(Value *OtherV,
                         llvm::function_ref<bool(const Use &)> ShouldReplace);
  void replaceAllUsesWith(Value *Other);

  /// \Returns the LLVM IR name of the bottom-most LLVM value.
  StringRef getName() const { return Val->getName(); }

#ifndef NDEBUG
  /// Should crash if there is something wrong with the instruction.
  virtual void verify() const = 0;
  /// Returns the unique id in the form 'SB<number>.' like 'SB1.'
  std::string getUid() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const sandboxir::Value &V) {
    V.dumpOS(OS);
    return OS;
  }
  virtual void dumpOS(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_VALUE_H
