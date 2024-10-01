//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarchy that resembles that of LLVM IR
// but is in the `sandboxir` namespace:
//
// namespace sandboxir {
//
// Value -+- Argument
//        |
//        +- BasicBlock
//        |
//        +- User ------+- Constant ------ Function
//                      |
//                      +- Instruction -+- BinaryOperator
//                                      |
//                                      +- BranchInst
//                                      |
//                                      +- CastInst --------+- AddrSpaceCastInst
//                                      |                   |
//                                      |                   +- BitCastInst
//                                      |                   |
//                                      |                   +- FPExtInst
//                                      |                   |
//                                      |                   +- FPToSIInst
//                                      |                   |
//                                      |                   +- FPToUIInst
//                                      |                   |
//                                      |                   +- FPTruncInst
//                                      |                   |
//                                      |                   +- IntToPtrInst
//                                      |                   |
//                                      |                   +- PtrToIntInst
//                                      |                   |
//                                      |                   +- SExtInst
//                                      |                   |
//                                      |                   +- SIToFPInst
//                                      |                   |
//                                      |                   +- TruncInst
//                                      |                   |
//                                      |                   +- UIToFPInst
//                                      |                   |
//                                      |                   +- ZExtInst
//                                      |
//                                      +- CallBase --------+- CallBrInst
//                                      |                   |
//                                      |                   +- CallInst
//                                      |                   |
//                                      |                   +- InvokeInst
//                                      |
//                                      +- CmpInst ---------+- ICmpInst
//                                      |                   |
//                                      |                   +- FCmpInst
//                                      |
//                                      +- ExtractElementInst
//                                      |
//                                      +- GetElementPtrInst
//                                      |
//                                      +- InsertElementInst
//                                      |
//                                      +- OpaqueInst
//                                      |
//                                      +- PHINode
//                                      |
//                                      +- ReturnInst
//                                      |
//                                      +- SelectInst
//                                      |
//                                      +- ShuffleVectorInst
//                                      |
//                                      +- ExtractValueInst
//                                      |
//                                      +- InsertValueInst
//                                      |
//                                      +- StoreInst
//                                      |
//                                      +- UnaryInstruction -+- LoadInst
//                                      |                    |
//                                      |                    +- CastInst
//                                      |
//                                      +- UnaryOperator
//                                      |
//                                      +- UnreachableInst
//
// Use
//
// } // namespace sandboxir
//

#ifndef LLVM_SANDBOXIR_SANDBOXIR_H
#define LLVM_SANDBOXIR_SANDBOXIR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Tracker.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/Use.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

namespace llvm {

namespace sandboxir {

class BasicBlock;
class ConstantInt;
class ConstantFP;
class ConstantAggregateZero;
class ConstantPointerNull;
class PoisonValue;
class BlockAddress;
class DSOLocalEquivalent;
class ConstantTokenNone;
class GlobalValue;
class GlobalObject;
class GlobalIFunc;
class GlobalVariable;
class GlobalAlias;
class NoCFIValue;
class ConstantPtrAuth;
class ConstantExpr;
class Context;
class Function;
class Module;
class Instruction;
class VAArgInst;
class FreezeInst;
class FenceInst;
class SelectInst;
class ExtractElementInst;
class InsertElementInst;
class ShuffleVectorInst;
class ExtractValueInst;
class InsertValueInst;
class BranchInst;
class UnaryInstruction;
class LoadInst;
class ReturnInst;
class StoreInst;
class User;
class UnreachableInst;
class Value;
class CallBase;
class CallInst;
class InvokeInst;
class CallBrInst;
class LandingPadInst;
class FuncletPadInst;
class CatchPadInst;
class CleanupPadInst;
class CatchReturnInst;
class CleanupReturnInst;
class GetElementPtrInst;
class CastInst;
class PossiblyNonNegInst;
class PtrToIntInst;
class BitCastInst;
class AllocaInst;
class ResumeInst;
class CatchSwitchInst;
class SwitchInst;
class UnaryOperator;
class BinaryOperator;
class PossiblyDisjointInst;
class AtomicRMWInst;
class AtomicCmpXchgInst;
class CmpInst;
class ICmpInst;
class FCmpInst;

/// Iterator for the `Use` edges of a User's operands.
/// \Returns the operand `Use` when dereferenced.
class OperandUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty OperandUseIterator.
  OperandUseIterator(const class Use &Use) : Use(Use) {}
  friend class User;                                  // For constructor
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  OperandUseIterator() = default;
  value_type operator*() const;
  OperandUseIterator &operator++();
  OperandUseIterator operator++(int) {
    auto Copy = *this;
    this->operator++();
    return Copy;
  }
  bool operator==(const OperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const OperandUseIterator &Other) const {
    return !(*this == Other);
  }
  OperandUseIterator operator+(unsigned Num) const;
  OperandUseIterator operator-(unsigned Num) const;
  int operator-(const OperandUseIterator &Other) const;
};

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
#include "llvm/SandboxIR/SandboxIRValues.def"
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
#include "llvm/SandboxIR/SandboxIRValues.def"
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

/// Argument of a sandboxir::Function.
class Argument : public sandboxir::Value {
  Argument(llvm::Argument *Arg, sandboxir::Context &Ctx)
      : sandboxir::Value(ClassID::Argument, Arg, Ctx) {}
  friend class Context; // For constructor.

public:
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Argument;
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Argument>(Val) && "Expected Argument!");
  }
  void printAsOperand(raw_ostream &OS) const;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

/// A sandboxir::User has operands.
class User : public Value {
protected:
  User(ClassID ID, llvm::Value *V, Context &Ctx) : Value(ID, V, Ctx) {}

  /// \Returns the Use edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  Use getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  /// \Returns the Use for the \p OpIdx'th operand. This is virtual to allow
  /// instructions to deviate from the LLVM IR operands, which is a requirement
  /// for sandboxir Instructions that consist of more than one LLVM Instruction.
  virtual Use getOperandUseInternal(unsigned OpIdx, bool Verify) const = 0;
  friend class OperandUseIterator; // for getOperandUseInternal()

  /// The default implementation works only for single-LLVMIR-instruction
  /// Users and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const Use &Use) const {
    return Use.LLVMUse->getOperandNo();
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const Use &Use) const = 0;
  friend unsigned Use::getOperandNo() const; // For getUseOperandNo()

  void swapOperandsInternal(unsigned OpIdxA, unsigned OpIdxB) {
    assert(OpIdxA < getNumOperands() && "OpIdxA out of bounds!");
    assert(OpIdxB < getNumOperands() && "OpIdxB out of bounds!");
    auto UseA = getOperandUse(OpIdxA);
    auto UseB = getOperandUse(OpIdxB);
    UseA.swap(UseB);
  }

#ifndef NDEBUG
  void verifyUserOfLLVMUse(const llvm::Use &Use) const;
#endif // NDEBUG

public:
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  using op_iterator = OperandUseIterator;
  using const_op_iterator = OperandUseIterator;
  using op_range = iterator_range<op_iterator>;
  using const_op_range = iterator_range<const_op_iterator>;

  virtual op_iterator op_begin() {
    assert(isa<llvm::User>(Val) && "Expect User value!");
    return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
  }
  virtual op_iterator op_end() {
    assert(isa<llvm::User>(Val) && "Expect User value!");
    return op_iterator(
        getOperandUseInternal(getNumOperands(), /*Verify=*/false));
  }
  virtual const_op_iterator op_begin() const {
    return const_cast<User *>(this)->op_begin();
  }
  virtual const_op_iterator op_end() const {
    return const_cast<User *>(this)->op_end();
  }

  op_range operands() { return make_range<op_iterator>(op_begin(), op_end()); }
  const_op_range operands() const {
    return make_range<const_op_iterator>(op_begin(), op_end());
  }
  Value *getOperand(unsigned OpIdx) const { return getOperandUse(OpIdx).get(); }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  Use getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  virtual unsigned getNumOperands() const {
    return isa<llvm::User>(Val) ? cast<llvm::User>(Val)->getNumOperands() : 0;
  }

  virtual void setOperand(unsigned OperandIdx, Value *Operand);
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  bool replaceUsesOfWith(Value *FromV, Value *ToV);

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::User>(Val) && "Expected User!");
  }
  void dumpCommonHeader(raw_ostream &OS) const final;
  void dumpOS(raw_ostream &OS) const override {
    // TODO: Remove this tmp implementation once we get the Instruction classes.
  }
#endif
};

class Constant : public sandboxir::User {
protected:
  Constant(llvm::Constant *C, sandboxir::Context &SBCtx)
      : sandboxir::User(ClassID::Constant, C, SBCtx) {}
  Constant(ClassID ID, llvm::Constant *C, sandboxir::Context &SBCtx)
      : sandboxir::User(ID, C, SBCtx) {}
  friend class ConstantInt; // For constructor.
  friend class Function;    // For constructor
  friend class Context;     // For constructor.
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const override {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
#define DEF_CONST(ID, CLASS) case ClassID::ID:
#include "llvm/SandboxIR/SandboxIRValues.def"
      return true;
    default:
      return false;
    }
  }
  sandboxir::Context &getParent() const { return getContext(); }
  unsigned getUseOperandNo(const Use &Use) const override {
    return getUseOperandNoDefault(Use);
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::Constant>(Val) && "Expected Constant!");
  }
  void dumpOS(raw_ostream &OS) const override;
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantInt : public Constant {
  ConstantInt(llvm::ConstantInt *C, Context &Ctx)
      : Constant(ClassID::ConstantInt, C, Ctx) {}
  friend class Context; // For constructor.

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    llvm_unreachable("ConstantInt has no operands!");
  }

public:
  static ConstantInt *getTrue(Context &Ctx);
  static ConstantInt *getFalse(Context &Ctx);
  static ConstantInt *getBool(Context &Ctx, bool V);
  static Constant *getTrue(Type *Ty);
  static Constant *getFalse(Type *Ty);
  static Constant *getBool(Type *Ty, bool V);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  static ConstantInt *get(Type *Ty, uint64_t V, bool IsSigned = false);

  /// Return a ConstantInt with the specified integer value for the specified
  /// type. If the type is wider than 64 bits, the value will be zero-extended
  /// to fit the type, unless IsSigned is true, in which case the value will
  /// be interpreted as a 64-bit signed integer and sign-extended to fit
  /// the type.
  /// Get a ConstantInt for a specific value.
  static ConstantInt *get(IntegerType *Ty, uint64_t V, bool IsSigned = false);

  /// Return a ConstantInt with the specified value for the specified type. The
  /// value V will be canonicalized to a an unsigned APInt. Accessing it with
  /// either getSExtValue() or getZExtValue() will yield a correctly sized and
  /// signed value for the type Ty.
  /// Get a ConstantInt for a specific signed value.
  static ConstantInt *getSigned(IntegerType *Ty, int64_t V);
  static Constant *getSigned(Type *Ty, int64_t V);

  /// Return a ConstantInt with the specified value and an implied Type. The
  /// type is the integer type that corresponds to the bit width of the value.
  static ConstantInt *get(Context &Ctx, const APInt &V);

  /// Return a ConstantInt constructed from the string strStart with the given
  /// radix.
  static ConstantInt *get(IntegerType *Ty, StringRef Str, uint8_t Radix);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  static Constant *get(Type *Ty, const APInt &V);

  /// Return the constant as an APInt value reference. This allows clients to
  /// obtain a full-precision copy of the value.
  /// Return the constant's value.
  inline const APInt &getValue() const {
    return cast<llvm::ConstantInt>(Val)->getValue();
  }

  /// getBitWidth - Return the scalar bitwidth of this constant.
  unsigned getBitWidth() const {
    return cast<llvm::ConstantInt>(Val)->getBitWidth();
  }
  /// Return the constant as a 64-bit unsigned integer value after it
  /// has been zero extended as appropriate for the type of this constant. Note
  /// that this method can assert if the value does not fit in 64 bits.
  /// Return the zero extended value.
  inline uint64_t getZExtValue() const {
    return cast<llvm::ConstantInt>(Val)->getZExtValue();
  }

  /// Return the constant as a 64-bit integer value after it has been sign
  /// extended as appropriate for the type of this constant. Note that
  /// this method can assert if the value does not fit in 64 bits.
  /// Return the sign extended value.
  inline int64_t getSExtValue() const {
    return cast<llvm::ConstantInt>(Val)->getSExtValue();
  }

  /// Return the constant as an llvm::MaybeAlign.
  /// Note that this method can assert if the value does not fit in 64 bits or
  /// is not a power of two.
  inline MaybeAlign getMaybeAlignValue() const {
    return cast<llvm::ConstantInt>(Val)->getMaybeAlignValue();
  }

  /// Return the constant as an llvm::Align, interpreting `0` as `Align(1)`.
  /// Note that this method can assert if the value does not fit in 64 bits or
  /// is not a power of two.
  inline Align getAlignValue() const {
    return cast<llvm::ConstantInt>(Val)->getAlignValue();
  }

  /// A helper method that can be used to determine if the constant contained
  /// within is equal to a constant.  This only works for very small values,
  /// because this is all that can be represented with all types.
  /// Determine if this constant's value is same as an unsigned char.
  bool equalsInt(uint64_t V) const {
    return cast<llvm::ConstantInt>(Val)->equalsInt(V);
  }

  /// Variant of the getType() method to always return an IntegerType, which
  /// reduces the amount of casting needed in parts of the compiler.
  IntegerType *getIntegerType() const;

  /// This static method returns true if the type Ty is big enough to
  /// represent the value V. This can be used to avoid having the get method
  /// assert when V is larger than Ty can represent. Note that there are two
  /// versions of this method, one for unsigned and one for signed integers.
  /// Although ConstantInt canonicalizes everything to an unsigned integer,
  /// the signed version avoids callers having to convert a signed quantity
  /// to the appropriate unsigned type before calling the method.
  /// @returns true if V is a valid value for type Ty
  /// Determine if the value is in range for the given type.
  static bool isValueValidForType(Type *Ty, uint64_t V);
  static bool isValueValidForType(Type *Ty, int64_t V);

  bool isNegative() const { return cast<llvm::ConstantInt>(Val)->isNegative(); }

  /// This is just a convenience method to make client code smaller for a
  /// common code. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  bool isZero() const { return cast<llvm::ConstantInt>(Val)->isZero(); }

  /// This is just a convenience method to make client code smaller for a
  /// common case. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  /// Determine if the value is one.
  bool isOne() const { return cast<llvm::ConstantInt>(Val)->isOne(); }

  /// This function will return true iff every bit in this constant is set
  /// to true.
  /// @returns true iff this constant's bits are all set to true.
  /// Determine if the value is all ones.
  bool isMinusOne() const { return cast<llvm::ConstantInt>(Val)->isMinusOne(); }

  /// This function will return true iff this constant represents the largest
  /// value that may be represented by the constant's type.
  /// @returns true iff this is the largest value that may be represented
  /// by this type.
  /// Determine if the value is maximal.
  bool isMaxValue(bool IsSigned) const {
    return cast<llvm::ConstantInt>(Val)->isMaxValue(IsSigned);
  }

  /// This function will return true iff this constant represents the smallest
  /// value that may be represented by this constant's type.
  /// @returns true if this is the smallest value that may be represented by
  /// this type.
  /// Determine if the value is minimal.
  bool isMinValue(bool IsSigned) const {
    return cast<llvm::ConstantInt>(Val)->isMinValue(IsSigned);
  }

  /// This function will return true iff this constant represents a value with
  /// active bits bigger than 64 bits or a value greater than the given uint64_t
  /// value.
  /// @returns true iff this constant is greater or equal to the given number.
  /// Determine if the value is greater or equal to the given number.
  bool uge(uint64_t Num) const {
    return cast<llvm::ConstantInt>(Val)->uge(Num);
  }

  /// getLimitedValue - If the value is smaller than the specified limit,
  /// return it, otherwise return the limit value.  This causes the value
  /// to saturate to the limit.
  /// @returns the min of the value of the constant and the specified value
  /// Get the constant's value with a saturation limit
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return cast<llvm::ConstantInt>(Val)->getLimitedValue(Limit);
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantInt;
  }
  unsigned getUseOperandNo(const Use &Use) const override {
    llvm_unreachable("ConstantInt has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantInt>(Val) && "Expected a ConstantInst!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantFP final : public Constant {
  ConstantFP(llvm::ConstantFP *C, Context &Ctx)
      : Constant(ClassID::ConstantFP, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// This returns a ConstantFP, or a vector containing a splat of a ConstantFP,
  /// for the specified value in the specified type. This should only be used
  /// for simple constant values like 2.0/1.0 etc, that are known-valid both as
  /// host double and as the target format.
  static Constant *get(Type *Ty, double V);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantFP for the given value.
  static Constant *get(Type *Ty, const APFloat &V);

  static Constant *get(Type *Ty, StringRef Str);

  static ConstantFP *get(const APFloat &V, Context &Ctx);

  static Constant *getNaN(Type *Ty, bool Negative = false,
                          uint64_t Payload = 0);
  static Constant *getQNaN(Type *Ty, bool Negative = false,
                           APInt *Payload = nullptr);
  static Constant *getSNaN(Type *Ty, bool Negative = false,
                           APInt *Payload = nullptr);
  static Constant *getZero(Type *Ty, bool Negative = false);

  static Constant *getNegativeZero(Type *Ty);
  static Constant *getInfinity(Type *Ty, bool Negative = false);

  /// Return true if Ty is big enough to represent V.
  static bool isValueValidForType(Type *Ty, const APFloat &V);

  inline const APFloat &getValueAPF() const {
    return cast<llvm::ConstantFP>(Val)->getValueAPF();
  }
  inline const APFloat &getValue() const {
    return cast<llvm::ConstantFP>(Val)->getValue();
  }

  /// Return true if the value is positive or negative zero.
  bool isZero() const { return cast<llvm::ConstantFP>(Val)->isZero(); }

  /// Return true if the sign bit is set.
  bool isNegative() const { return cast<llvm::ConstantFP>(Val)->isNegative(); }

  /// Return true if the value is infinity
  bool isInfinity() const { return cast<llvm::ConstantFP>(Val)->isInfinity(); }

  /// Return true if the value is a NaN.
  bool isNaN() const { return cast<llvm::ConstantFP>(Val)->isNaN(); }

  /// We don't rely on operator== working on double values, as it returns true
  /// for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.  The version with a double operand is retained
  /// because it's so convenient to write isExactlyValue(2.0), but please use
  /// it only for simple constants.
  bool isExactlyValue(const APFloat &V) const {
    return cast<llvm::ConstantFP>(Val)->isExactlyValue(V);
  }

  bool isExactlyValue(double V) const {
    return cast<llvm::ConstantFP>(Val)->isExactlyValue(V);
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantFP;
  }

  // TODO: Better name: getOperandNo(const Use&). Should be private.
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantFP has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantFP>(Val) && "Expected a ConstantFP!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

/// Base class for aggregate constants (with operands).
class ConstantAggregate : public Constant {
protected:
  ConstantAggregate(ClassID ID, llvm::Constant *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    auto ID = From->getSubclassID();
    return ID == ClassID::ConstantVector || ID == ClassID::ConstantStruct ||
           ID == ClassID::ConstantArray;
  }
};

class ConstantArray final : public ConstantAggregate {
  ConstantArray(llvm::ConstantArray *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantArray, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static Constant *get(ArrayType *T, ArrayRef<Constant *> V);
  ArrayType *getType() const;

  // TODO: Missing functions: getType(), getTypeForElements(), getAnon(), get().

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantArray;
  }
};

class ConstantStruct final : public ConstantAggregate {
  ConstantStruct(llvm::ConstantStruct *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantStruct, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static Constant *get(StructType *T, ArrayRef<Constant *> V);

  template <typename... Csts>
  static std::enable_if_t<are_base_of<Constant, Csts...>::value, Constant *>
  get(StructType *T, Csts *...Vs) {
    return get(T, ArrayRef<Constant *>({Vs...}));
  }
  /// Return an anonymous struct that has the specified elements.
  /// If the struct is possibly empty, then you must specify a context.
  static Constant *getAnon(ArrayRef<Constant *> V, bool Packed = false) {
    return get(getTypeForElements(V, Packed), V);
  }
  static Constant *getAnon(Context &Ctx, ArrayRef<Constant *> V,
                           bool Packed = false) {
    return get(getTypeForElements(Ctx, V, Packed), V);
  }
  /// This version of the method allows an empty list.
  static StructType *getTypeForElements(Context &Ctx, ArrayRef<Constant *> V,
                                        bool Packed = false);
  /// Return an anonymous struct type to use for a constant with the specified
  /// set of elements. The list must not be empty.
  static StructType *getTypeForElements(ArrayRef<Constant *> V,
                                        bool Packed = false) {
    assert(!V.empty() &&
           "ConstantStruct::getTypeForElements cannot be called on empty list");
    return getTypeForElements(V[0]->getContext(), V, Packed);
  }

  /// Specialization - reduce amount of casting.
  inline StructType *getType() const {
    return cast<StructType>(Value::getType());
  }

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantStruct;
  }
};

class ConstantVector final : public ConstantAggregate {
  ConstantVector(llvm::ConstantVector *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantVector, C, Ctx) {}
  friend class Context; // For constructor.

public:
  // TODO: Missing functions: getSplat(), getType(), getSplatValue(), get().

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantVector;
  }
};

// TODO: Inherit from ConstantData.
class ConstantAggregateZero final : public Constant {
  ConstantAggregateZero(llvm::ConstantAggregateZero *C, Context &Ctx)
      : Constant(ClassID::ConstantAggregateZero, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static ConstantAggregateZero *get(Type *Ty);
  /// If this CAZ has array or vector type, return a zero with the right element
  /// type.
  Constant *getSequentialElement() const;
  /// If this CAZ has struct type, return a zero with the right element type for
  /// the specified element.
  Constant *getStructElement(unsigned Elt) const;
  /// Return a zero of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  Constant *getElementValue(Constant *C) const;
  /// Return a zero of the right value for the specified GEP index.
  Constant *getElementValue(unsigned Idx) const;
  /// Return the number of elements in the array, vector, or struct.
  ElementCount getElementCount() const {
    return cast<llvm::ConstantAggregateZero>(Val)->getElementCount();
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantAggregateZero;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantAggregateZero has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantAggregateZero>(Val) && "Expected a CAZ!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: Inherit from ConstantData.
class ConstantPointerNull final : public Constant {
  ConstantPointerNull(llvm::ConstantPointerNull *C, Context &Ctx)
      : Constant(ClassID::ConstantPointerNull, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static ConstantPointerNull *get(PointerType *Ty);

  PointerType *getType() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantPointerNull;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantPointerNull has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantPointerNull>(Val) && "Expected a CPNull!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: Inherit from ConstantData.
class UndefValue : public Constant {
protected:
  UndefValue(llvm::UndefValue *C, Context &Ctx)
      : Constant(ClassID::UndefValue, C, Ctx) {}
  UndefValue(ClassID ID, llvm::Constant *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Static factory methods - Return an 'undef' object of the specified type.
  static UndefValue *get(Type *T);

  /// If this Undef has array or vector type, return a undef with the right
  /// element type.
  UndefValue *getSequentialElement() const;

  /// If this undef has struct type, return a undef with the right element type
  /// for the specified element.
  UndefValue *getStructElement(unsigned Elt) const;

  /// Return an undef of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  UndefValue *getElementValue(Constant *C) const;

  /// Return an undef of the right value for the specified GEP index.
  UndefValue *getElementValue(unsigned Idx) const;

  /// Return the number of elements in the array, vector, or struct.
  unsigned getNumElements() const {
    return cast<llvm::UndefValue>(Val)->getNumElements();
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::UndefValue ||
           From->getSubclassID() == ClassID::PoisonValue;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("UndefValue has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::UndefValue>(Val) && "Expected an UndefValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class PoisonValue final : public UndefValue {
  PoisonValue(llvm::PoisonValue *C, Context &Ctx)
      : UndefValue(ClassID::PoisonValue, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Static factory methods - Return an 'poison' object of the specified type.
  static PoisonValue *get(Type *T);

  /// If this poison has array or vector type, return a poison with the right
  /// element type.
  PoisonValue *getSequentialElement() const;

  /// If this poison has struct type, return a poison with the right element
  /// type for the specified element.
  PoisonValue *getStructElement(unsigned Elt) const;

  /// Return an poison of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  PoisonValue *getElementValue(Constant *C) const;

  /// Return an poison of the right value for the specified GEP index.
  PoisonValue *getElementValue(unsigned Idx) const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::PoisonValue;
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::PoisonValue>(Val) && "Expected a PoisonValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalValue : public Constant {
protected:
  GlobalValue(ClassID ID, llvm::GlobalValue *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}
  friend class Context; // For constructor.

public:
  using LinkageTypes = llvm::GlobalValue::LinkageTypes;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
    case ClassID::Function:
    case ClassID::GlobalVariable:
    case ClassID::GlobalAlias:
    case ClassID::GlobalIFunc:
      return true;
    default:
      return false;
    }
  }

  unsigned getAddressSpace() const {
    return cast<llvm::GlobalValue>(Val)->getAddressSpace();
  }
  bool hasGlobalUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->hasGlobalUnnamedAddr();
  }

  /// Returns true if this value's address is not significant in this module.
  /// This attribute is intended to be used only by the code generator and LTO
  /// to allow the linker to decide whether the global needs to be in the symbol
  /// table. It should probably not be used in optimizations, as the value may
  /// have uses outside the module; use hasGlobalUnnamedAddr() instead.
  bool hasAtLeastLocalUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->hasAtLeastLocalUnnamedAddr();
  }

  using UnnamedAddr = llvm::GlobalValue::UnnamedAddr;

  UnnamedAddr getUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->getUnnamedAddr();
  }
  void setUnnamedAddr(UnnamedAddr V);

  static UnnamedAddr getMinUnnamedAddr(UnnamedAddr A, UnnamedAddr B) {
    return llvm::GlobalValue::getMinUnnamedAddr(A, B);
  }

  bool hasComdat() const { return cast<llvm::GlobalValue>(Val)->hasComdat(); }

  // TODO: We need a SandboxIR Comdat if we want to implement getComdat().
  using VisibilityTypes = llvm::GlobalValue::VisibilityTypes;
  VisibilityTypes getVisibility() const {
    return cast<llvm::GlobalValue>(Val)->getVisibility();
  }
  bool hasDefaultVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasDefaultVisibility();
  }
  bool hasHiddenVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasHiddenVisibility();
  }
  bool hasProtectedVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasProtectedVisibility();
  }
  void setVisibility(VisibilityTypes V);

  // TODO: Add missing functions.
};

class GlobalObject : public GlobalValue {
protected:
  GlobalObject(ClassID ID, llvm::GlobalObject *C, Context &Ctx)
      : GlobalValue(ID, C, Ctx) {}
  friend class Context; // For constructor.
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
    case ClassID::Function:
    case ClassID::GlobalVariable:
    case ClassID::GlobalIFunc:
      return true;
    default:
      return false;
    }
  }

  /// FIXME: Remove this function once transition to Align is over.
  uint64_t getAlignment() const {
    return cast<llvm::GlobalObject>(Val)->getAlignment();
  }

  /// Returns the alignment of the given variable or function.
  ///
  /// Note that for functions this is the alignment of the code, not the
  /// alignment of a function pointer.
  MaybeAlign getAlign() const {
    return cast<llvm::GlobalObject>(Val)->getAlign();
  }

  // TODO: Add missing: setAlignment(Align)

  /// Sets the alignment attribute of the GlobalObject.
  /// This method will be deprecated as the alignment property should always be
  /// defined.
  void setAlignment(MaybeAlign Align);

  unsigned getGlobalObjectSubClassData() const {
    return cast<llvm::GlobalObject>(Val)->getGlobalObjectSubClassData();
  }

  void setGlobalObjectSubClassData(unsigned V);

  /// Check if this global has a custom object file section.
  ///
  /// This is more efficient than calling getSection() and checking for an empty
  /// string.
  bool hasSection() const {
    return cast<llvm::GlobalObject>(Val)->hasSection();
  }

  /// Get the custom section of this global if it has one.
  ///
  /// If this global does not have a custom section, this will be empty and the
  /// default object file section (.text, .data, etc) will be used.
  StringRef getSection() const {
    return cast<llvm::GlobalObject>(Val)->getSection();
  }

  /// Change the section for this global.
  ///
  /// Setting the section to the empty string tells LLVM to choose an
  /// appropriate default object file section.
  void setSection(StringRef S);

  bool hasComdat() const { return cast<llvm::GlobalObject>(Val)->hasComdat(); }

  // TODO: implement get/setComdat(), etc. once we have a sandboxir::Comdat.

  // TODO: We currently don't support Metadata in sandboxir so all
  // Metadata-related functions are missing.

  using VCallVisibility = llvm::GlobalObject::VCallVisibility;

  VCallVisibility getVCallVisibility() const {
    return cast<llvm::GlobalObject>(Val)->getVCallVisibility();
  }

  /// Returns true if the alignment of the value can be unilaterally
  /// increased.
  ///
  /// Note that for functions this is the alignment of the code, not the
  /// alignment of a function pointer.
  bool canIncreaseAlignment() const {
    return cast<llvm::GlobalObject>(Val)->canIncreaseAlignment();
  }
};

/// Provides API functions, like getIterator() and getReverseIterator() to
/// GlobalIFunc, Function, GlobalVariable and GlobalAlias. In LLVM IR these are
/// provided by ilist_node.
template <typename GlobalT, typename LLVMGlobalT, typename ParentT,
          typename LLVMParentT>
class GlobalWithNodeAPI : public ParentT {
  /// Helper for mapped_iterator.
  struct LLVMGVToGV {
    Context &Ctx;
    LLVMGVToGV(Context &Ctx) : Ctx(Ctx) {}
    GlobalT &operator()(LLVMGlobalT &LLVMGV) const;
  };

public:
  GlobalWithNodeAPI(Value::ClassID ID, LLVMParentT *C, Context &Ctx)
      : ParentT(ID, C, Ctx) {}

  Module *getParent() const {
    llvm::Module *LLVMM = cast<LLVMGlobalT>(this->Val)->getParent();
    return this->Ctx.getModule(LLVMM);
  }

  using iterator = mapped_iterator<
      decltype(static_cast<LLVMGlobalT *>(nullptr)->getIterator()), LLVMGVToGV>;
  using reverse_iterator = mapped_iterator<
      decltype(static_cast<LLVMGlobalT *>(nullptr)->getReverseIterator()),
      LLVMGVToGV>;
  iterator getIterator() const {
    auto *LLVMGV = cast<LLVMGlobalT>(this->Val);
    LLVMGVToGV ToGV(this->Ctx);
    return map_iterator(LLVMGV->getIterator(), ToGV);
  }
  reverse_iterator getReverseIterator() const {
    auto *LLVMGV = cast<LLVMGlobalT>(this->Val);
    LLVMGVToGV ToGV(this->Ctx);
    return map_iterator(LLVMGV->getReverseIterator(), ToGV);
  }
};

class GlobalIFunc final
    : public GlobalWithNodeAPI<GlobalIFunc, llvm::GlobalIFunc, GlobalObject,
                               llvm::GlobalObject> {
  GlobalIFunc(llvm::GlobalObject *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalIFunc, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalIFunc;
  }

  // TODO: Missing create() because we don't have a sandboxir::Module yet.

  // TODO: Missing functions: copyAttributesFrom(), removeFromParent(),
  // eraseFromParent()

  void setResolver(Constant *Resolver);

  Constant *getResolver() const;

  // Return the resolver function after peeling off potential ConstantExpr
  // indirection.
  Function *getResolverFunction();
  const Function *getResolverFunction() const {
    return const_cast<GlobalIFunc *>(this)->getResolverFunction();
  }

  static bool isValidLinkage(LinkageTypes L) {
    return llvm::GlobalIFunc::isValidLinkage(L);
  }

  // TODO: Missing applyAlongResolverPath().

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::GlobalIFunc>(Val) && "Expected a GlobalIFunc!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalVariable final
    : public GlobalWithNodeAPI<GlobalVariable, llvm::GlobalVariable,
                               GlobalObject, llvm::GlobalObject> {
  GlobalVariable(llvm::GlobalObject *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalVariable, C, Ctx) {}
  friend class Context; // For constructor.

  /// Helper for mapped_iterator.
  struct LLVMGVToGV {
    Context &Ctx;
    LLVMGVToGV(Context &Ctx) : Ctx(Ctx) {}
    GlobalVariable &operator()(llvm::GlobalVariable &LLVMGV) const;
  };

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalVariable;
  }

  /// Definitions have initializers, declarations don't.
  ///
  inline bool hasInitializer() const {
    return cast<llvm::GlobalVariable>(Val)->hasInitializer();
  }

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
    return cast<llvm::GlobalVariable>(Val)->hasDefinitiveInitializer();
  }

  /// hasUniqueInitializer - Whether the global variable has an initializer, and
  /// any changes made to the initializer will turn up in the final executable.
  inline bool hasUniqueInitializer() const {
    return cast<llvm::GlobalVariable>(Val)->hasUniqueInitializer();
  }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  Constant *getInitializer() const;
  /// setInitializer - Sets the initializer for this global variable, removing
  /// any existing initializer if InitVal==NULL. The initializer must have the
  /// type getValueType().
  void setInitializer(Constant *InitVal);

  // TODO: Add missing replaceInitializer(). Requires special tracker

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const {
    return cast<llvm::GlobalVariable>(Val)->isConstant();
  }
  void setConstant(bool V);

  bool isExternallyInitialized() const {
    return cast<llvm::GlobalVariable>(Val)->isExternallyInitialized();
  }
  void setExternallyInitialized(bool Val);

  // TODO: Missing copyAttributesFrom()

  // TODO: Missing removeFromParent(), eraseFromParent(), dropAllReferences()

  // TODO: Missing addDebugInfo(), getDebugInfo()

  // TODO: Missing attribute setter functions: addAttribute(), setAttributes().
  //       There seems to be no removeAttribute() so we can't undo them.

  /// Return true if the attribute exists.
  bool hasAttribute(Attribute::AttrKind Kind) const {
    return cast<llvm::GlobalVariable>(Val)->hasAttribute(Kind);
  }

  /// Return true if the attribute exists.
  bool hasAttribute(StringRef Kind) const {
    return cast<llvm::GlobalVariable>(Val)->hasAttribute(Kind);
  }

  /// Return true if any attributes exist.
  bool hasAttributes() const {
    return cast<llvm::GlobalVariable>(Val)->hasAttributes();
  }

  /// Return the attribute object.
  Attribute getAttribute(Attribute::AttrKind Kind) const {
    return cast<llvm::GlobalVariable>(Val)->getAttribute(Kind);
  }

  /// Return the attribute object.
  Attribute getAttribute(StringRef Kind) const {
    return cast<llvm::GlobalVariable>(Val)->getAttribute(Kind);
  }

  /// Return the attribute set for this global
  AttributeSet getAttributes() const {
    return cast<llvm::GlobalVariable>(Val)->getAttributes();
  }

  /// Return attribute set as list with index.
  /// FIXME: This may not be required once ValueEnumerators
  /// in bitcode-writer can enumerate attribute-set.
  AttributeList getAttributesAsList(unsigned Index) const {
    return cast<llvm::GlobalVariable>(Val)->getAttributesAsList(Index);
  }

  /// Check if section name is present
  bool hasImplicitSection() const {
    return cast<llvm::GlobalVariable>(Val)->hasImplicitSection();
  }

  /// Get the custom code model raw value of this global.
  ///
  unsigned getCodeModelRaw() const {
    return cast<llvm::GlobalVariable>(Val)->getCodeModelRaw();
  }

  /// Get the custom code model of this global if it has one.
  ///
  /// If this global does not have a custom code model, the empty instance
  /// will be returned.
  std::optional<CodeModel::Model> getCodeModel() const {
    return cast<llvm::GlobalVariable>(Val)->getCodeModel();
  }

  // TODO: Missing setCodeModel(). Requires custom tracker.

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::GlobalVariable>(Val) && "Expected a GlobalVariable!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalAlias final
    : public GlobalWithNodeAPI<GlobalAlias, llvm::GlobalAlias, GlobalValue,
                               llvm::GlobalValue> {
  GlobalAlias(llvm::GlobalAlias *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalAlias, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalAlias;
  }

  // TODO: Missing create() due to unimplemented sandboxir::Module.

  // TODO: Missing copyAttributresFrom().
  // TODO: Missing removeFromParent(), eraseFromParent().

  void setAliasee(Constant *Aliasee);
  Constant *getAliasee() const;

  const GlobalObject *getAliaseeObject() const;
  GlobalObject *getAliaseeObject() {
    return const_cast<GlobalObject *>(
        static_cast<const GlobalAlias *>(this)->getAliaseeObject());
  }

  static bool isValidLinkage(LinkageTypes L) {
    return llvm::GlobalAlias::isValidLinkage(L);
  }
};

class NoCFIValue final : public Constant {
  NoCFIValue(llvm::NoCFIValue *C, Context &Ctx)
      : Constant(ClassID::NoCFIValue, C, Ctx) {}
  friend class Context; // For constructor.

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  /// Return a NoCFIValue for the specified function.
  static NoCFIValue *get(GlobalValue *GV);

  GlobalValue *getGlobalValue() const;

  /// NoCFIValue is always a pointer.
  PointerType *getType() const;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::NoCFIValue;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::NoCFIValue>(Val) && "Expected a NoCFIValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class ConstantPtrAuth final : public Constant {
  ConstantPtrAuth(llvm::ConstantPtrAuth *C, Context &Ctx)
      : Constant(ClassID::ConstantPtrAuth, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a pointer signed with the specified parameters.
  static ConstantPtrAuth *get(Constant *Ptr, ConstantInt *Key,
                              ConstantInt *Disc, Constant *AddrDisc);
  /// The pointer that is signed in this ptrauth signed pointer.
  Constant *getPointer() const;

  /// The Key ID, an i32 constant.
  ConstantInt *getKey() const;

  /// The integer discriminator, an i64 constant, or 0.
  ConstantInt *getDiscriminator() const;

  /// The address discriminator if any, or the null constant.
  /// If present, this must be a value equivalent to the storage location of
  /// the only global-initializer user of the ptrauth signed pointer.
  Constant *getAddrDiscriminator() const;

  /// Whether there is any non-null address discriminator.
  bool hasAddressDiscriminator() const {
    return cast<llvm::ConstantPtrAuth>(Val)->hasAddressDiscriminator();
  }

  /// Whether the address uses a special address discriminator.
  /// These discriminators can't be used in real pointer-auth values; they
  /// can only be used in "prototype" values that indicate how some real
  /// schema is supposed to be produced.
  bool hasSpecialAddressDiscriminator(uint64_t Value) const {
    return cast<llvm::ConstantPtrAuth>(Val)->hasSpecialAddressDiscriminator(
        Value);
  }

  /// Check whether an authentication operation with key \p Key and (possibly
  /// blended) discriminator \p Discriminator is known to be compatible with
  /// this ptrauth signed pointer.
  bool isKnownCompatibleWith(const Value *Key, const Value *Discriminator,
                             const DataLayout &DL) const {
    return cast<llvm::ConstantPtrAuth>(Val)->isKnownCompatibleWith(
        Key->Val, Discriminator->Val, DL);
  }

  /// Produce a new ptrauth expression signing the given value using
  /// the same schema as is stored in one.
  ConstantPtrAuth *getWithSameSchema(Constant *Pointer) const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantPtrAuth;
  }
};

class ConstantExpr : public Constant {
  ConstantExpr(llvm::ConstantExpr *C, Context &Ctx)
      : Constant(ClassID::ConstantExpr, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantExpr;
  }
  // TODO: Missing functions.
};

class BlockAddress final : public Constant {
  BlockAddress(llvm::BlockAddress *C, Context &Ctx)
      : Constant(ClassID::BlockAddress, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a BlockAddress for the specified function and basic block.
  static BlockAddress *get(Function *F, BasicBlock *BB);

  /// Return a BlockAddress for the specified basic block.  The basic
  /// block must be embedded into a function.
  static BlockAddress *get(BasicBlock *BB);

  /// Lookup an existing \c BlockAddress constant for the given BasicBlock.
  ///
  /// \returns 0 if \c !BB->hasAddressTaken(), otherwise the \c BlockAddress.
  static BlockAddress *lookup(const BasicBlock *BB);

  Function *getFunction() const;
  BasicBlock *getBasicBlock() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::BlockAddress;
  }
};

class DSOLocalEquivalent final : public Constant {
  DSOLocalEquivalent(llvm::DSOLocalEquivalent *C, Context &Ctx)
      : Constant(ClassID::DSOLocalEquivalent, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a DSOLocalEquivalent for the specified global value.
  static DSOLocalEquivalent *get(GlobalValue *GV);

  GlobalValue *getGlobalValue() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::DSOLocalEquivalent;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("DSOLocalEquivalent has no operands!");
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::DSOLocalEquivalent>(Val) &&
           "Expected a DSOLocalEquivalent!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantTokenNone final : public Constant {
  ConstantTokenNone(llvm::ConstantTokenNone *C, Context &Ctx)
      : Constant(ClassID::ConstantTokenNone, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return the ConstantTokenNone.
  static ConstantTokenNone *get(Context &Ctx);

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantTokenNone;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantTokenNone has no operands!");
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantTokenNone>(Val) &&
           "Expected a ConstantTokenNone!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

/// Iterator for `Instruction`s in a `BasicBlock.
/// \Returns an sandboxir::Instruction & when derereferenced.
class BBIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Instruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  llvm::BasicBlock *BB;
  llvm::BasicBlock::iterator It;
  Context *Ctx;
  pointer getInstr(llvm::BasicBlock::iterator It) const;

public:
  BBIterator() : BB(nullptr), Ctx(nullptr) {}
  BBIterator(llvm::BasicBlock *BB, llvm::BasicBlock::iterator It, Context *Ctx)
      : BB(BB), It(It), Ctx(Ctx) {}
  reference operator*() const { return *getInstr(It); }
  BBIterator &operator++();
  BBIterator operator++(int) {
    auto Copy = *this;
    ++*this;
    return Copy;
  }
  BBIterator &operator--();
  BBIterator operator--(int) {
    auto Copy = *this;
    --*this;
    return Copy;
  }
  bool operator==(const BBIterator &Other) const {
    assert(Ctx == Other.Ctx && "BBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const BBIterator &Other) const { return !(*this == Other); }
  /// \Returns the SBInstruction that corresponds to this iterator, or null if
  /// the instruction is not found in the IR-to-SandboxIR tables.
  pointer get() const { return getInstr(It); }
  /// \Returns the parent BB.
  BasicBlock *getNodeParent() const;
};

/// Contains a list of sandboxir::Instruction's.
class BasicBlock : public Value {
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildBasicBlockFromLLVMIR(llvm::BasicBlock *LLVMBB);
  friend class Context;     // For `buildBasicBlockFromIR`
  friend class Instruction; // For LLVM Val.

  BasicBlock(llvm::BasicBlock *BB, Context &SBCtx)
      : Value(ClassID::Block, BB, SBCtx) {
    buildBasicBlockFromLLVMIR(BB);
  }

public:
  ~BasicBlock() = default;
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == Value::ClassID::Block;
  }
  Function *getParent() const;
  using iterator = BBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<llvm::BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctx);
  }
  std::reverse_iterator<iterator> rbegin() const {
    return std::make_reverse_iterator(end());
  }
  std::reverse_iterator<iterator> rend() const {
    return std::make_reverse_iterator(begin());
  }
  Context &getContext() const { return Ctx; }
  Instruction *getTerminator() const;
  bool empty() const { return begin() == end(); }
  Instruction &front() const;
  Instruction &back() const;

#ifndef NDEBUG
  void verify() const final;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

/// A sandboxir::User with operands, opcode and linked with previous/next
/// instructions in an instruction list.
class Instruction : public sandboxir::User {
public:
  enum class Opcode {
#define OP(OPC) OPC,
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

protected:
  Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I,
              sandboxir::Context &SBCtx)
      : sandboxir::User(ID, I, SBCtx), Opc(Opc) {}

  Opcode Opc;

  /// A SandboxIR Instruction may map to multiple LLVM IR Instruction. This
  /// returns its topmost LLVM IR instruction.
  llvm::Instruction *getTopmostLLVMInstruction() const;
  friend class VAArgInst;          // For getTopmostLLVMInstruction().
  friend class FreezeInst;         // For getTopmostLLVMInstruction().
  friend class FenceInst;          // For getTopmostLLVMInstruction().
  friend class SelectInst;         // For getTopmostLLVMInstruction().
  friend class ExtractElementInst; // For getTopmostLLVMInstruction().
  friend class InsertElementInst;  // For getTopmostLLVMInstruction().
  friend class ShuffleVectorInst;  // For getTopmostLLVMInstruction().
  friend class ExtractValueInst;   // For getTopmostLLVMInstruction().
  friend class InsertValueInst;    // For getTopmostLLVMInstruction().
  friend class BranchInst;         // For getTopmostLLVMInstruction().
  friend class LoadInst;           // For getTopmostLLVMInstruction().
  friend class StoreInst;          // For getTopmostLLVMInstruction().
  friend class ReturnInst;         // For getTopmostLLVMInstruction().
  friend class CallInst;           // For getTopmostLLVMInstruction().
  friend class InvokeInst;         // For getTopmostLLVMInstruction().
  friend class CallBrInst;         // For getTopmostLLVMInstruction().
  friend class LandingPadInst;     // For getTopmostLLVMInstruction().
  friend class CatchPadInst;       // For getTopmostLLVMInstruction().
  friend class CleanupPadInst;     // For getTopmostLLVMInstruction().
  friend class CatchReturnInst;    // For getTopmostLLVMInstruction().
  friend class CleanupReturnInst;  // For getTopmostLLVMInstruction().
  friend class GetElementPtrInst;  // For getTopmostLLVMInstruction().
  friend class ResumeInst;         // For getTopmostLLVMInstruction().
  friend class CatchSwitchInst;    // For getTopmostLLVMInstruction().
  friend class SwitchInst;         // For getTopmostLLVMInstruction().
  friend class UnaryOperator;      // For getTopmostLLVMInstruction().
  friend class BinaryOperator;     // For getTopmostLLVMInstruction().
  friend class AtomicRMWInst;      // For getTopmostLLVMInstruction().
  friend class AtomicCmpXchgInst;  // For getTopmostLLVMInstruction().
  friend class AllocaInst;         // For getTopmostLLVMInstruction().
  friend class CastInst;           // For getTopmostLLVMInstruction().
  friend class PHINode;            // For getTopmostLLVMInstruction().
  friend class UnreachableInst;    // For getTopmostLLVMInstruction().
  friend class CmpInst;            // For getTopmostLLVMInstruction().

  /// \Returns the LLVM IR Instructions that this SandboxIR maps to in program
  /// order.
  virtual SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const = 0;
  friend class EraseFromParent; // For getLLVMInstrs().

public:
  static const char *getOpcodeName(Opcode Opc);
  /// This is used by BasicBlock::iterator.
  virtual unsigned getNumOfIRInstrs() const = 0;
  /// \Returns a BasicBlock::iterator for this Instruction.
  BBIterator getIterator() const;
  /// \Returns the next sandboxir::Instruction in the block, or nullptr if at
  /// the end of the block.
  Instruction *getNextNode() const;
  /// \Returns the previous sandboxir::Instruction in the block, or nullptr if
  /// at the beginning of the block.
  Instruction *getPrevNode() const;
  /// \Returns this Instruction's opcode. Note that SandboxIR has its own opcode
  /// state to allow for new SandboxIR-specific instructions.
  Opcode getOpcode() const { return Opc; }

  const char *getOpcodeName() const { return getOpcodeName(Opc); }

  // Note that these functions below are calling into llvm::Instruction.
  // A sandbox IR instruction could introduce a new opcode that could change the
  // behavior of one of these functions. It is better that these functions are
  // only added as needed and new sandbox IR instructions must explicitly check
  // if any of these functions could have a different behavior.

  bool isTerminator() const {
    return cast<llvm::Instruction>(Val)->isTerminator();
  }
  bool isUnaryOp() const { return cast<llvm::Instruction>(Val)->isUnaryOp(); }
  bool isBinaryOp() const { return cast<llvm::Instruction>(Val)->isBinaryOp(); }
  bool isIntDivRem() const {
    return cast<llvm::Instruction>(Val)->isIntDivRem();
  }
  bool isShift() const { return cast<llvm::Instruction>(Val)->isShift(); }
  bool isCast() const { return cast<llvm::Instruction>(Val)->isCast(); }
  bool isFuncletPad() const {
    return cast<llvm::Instruction>(Val)->isFuncletPad();
  }
  bool isSpecialTerminator() const {
    return cast<llvm::Instruction>(Val)->isSpecialTerminator();
  }
  bool isOnlyUserOfAnyOperand() const {
    return cast<llvm::Instruction>(Val)->isOnlyUserOfAnyOperand();
  }
  bool isLogicalShift() const {
    return cast<llvm::Instruction>(Val)->isLogicalShift();
  }

  //===--------------------------------------------------------------------===//
  // Metadata manipulation.
  //===--------------------------------------------------------------------===//

  /// Return true if the instruction has any metadata attached to it.
  bool hasMetadata() const {
    return cast<llvm::Instruction>(Val)->hasMetadata();
  }

  /// Return true if this instruction has metadata attached to it other than a
  /// debug location.
  bool hasMetadataOtherThanDebugLoc() const {
    return cast<llvm::Instruction>(Val)->hasMetadataOtherThanDebugLoc();
  }

  /// Return true if this instruction has the given type of metadata attached.
  bool hasMetadata(unsigned KindID) const {
    return cast<llvm::Instruction>(Val)->hasMetadata(KindID);
  }

  // TODO: Implement getMetadata and getAllMetadata after sandboxir::MDNode is
  // available.

  // TODO: More missing functions

  /// Detach this from its parent BasicBlock without deleting it.
  void removeFromParent();
  /// Detach this Value from its parent and delete it.
  void eraseFromParent();
  /// Insert this detached instruction before \p BeforeI.
  void insertBefore(Instruction *BeforeI);
  /// Insert this detached instruction after \p AfterI.
  void insertAfter(Instruction *AfterI);
  /// Insert this detached instruction into \p BB at \p WhereIt.
  void insertInto(BasicBlock *BB, const BBIterator &WhereIt);
  /// Move this instruction to \p WhereIt.
  void moveBefore(BasicBlock &BB, const BBIterator &WhereIt);
  /// Move this instruction before \p Before.
  void moveBefore(Instruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  /// Move this instruction after \p After.
  void moveAfter(Instruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  // TODO: This currently relies on LLVM IR Instruction::comesBefore which is
  // can be linear-time.
  /// Given an instruction Other in the same basic block as this instruction,
  /// return true if this instruction comes before Other.
  bool comesBefore(const Instruction *Other) const {
    return cast<llvm::Instruction>(Val)->comesBefore(
        cast<llvm::Instruction>(Other->Val));
  }
  /// \Returns the BasicBlock containing this Instruction, or null if it is
  /// detached.
  BasicBlock *getParent() const;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);

  /// Determine whether the no signed wrap flag is set.
  bool hasNoUnsignedWrap() const {
    return cast<llvm::Instruction>(Val)->hasNoUnsignedWrap();
  }
  /// Set or clear the nuw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoUnsignedWrap(bool B = true);
  /// Determine whether the no signed wrap flag is set.
  bool hasNoSignedWrap() const {
    return cast<llvm::Instruction>(Val)->hasNoSignedWrap();
  }
  /// Set or clear the nsw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoSignedWrap(bool B = true);
  /// Determine whether all fast-math-flags are set.
  bool isFast() const { return cast<llvm::Instruction>(Val)->isFast(); }
  /// Set or clear all fast-math-flags on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setFast(bool B);
  /// Determine whether the allow-reassociation flag is set.
  bool hasAllowReassoc() const {
    return cast<llvm::Instruction>(Val)->hasAllowReassoc();
  }
  /// Set or clear the reassociation flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReassoc(bool B);
  /// Determine whether the exact flag is set.
  bool isExact() const { return cast<llvm::Instruction>(Val)->isExact(); }
  /// Set or clear the exact flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setIsExact(bool B = true);
  /// Determine whether the no-NaNs flag is set.
  bool hasNoNaNs() const { return cast<llvm::Instruction>(Val)->hasNoNaNs(); }
  /// Set or clear the no-nans flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoNaNs(bool B);
  /// Determine whether the no-infs flag is set.
  bool hasNoInfs() const { return cast<llvm::Instruction>(Val)->hasNoInfs(); }
  /// Set or clear the no-infs flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoInfs(bool B);
  /// Determine whether the no-signed-zeros flag is set.
  bool hasNoSignedZeros() const {
    return cast<llvm::Instruction>(Val)->hasNoSignedZeros();
  }
  /// Set or clear the no-signed-zeros flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoSignedZeros(bool B);
  /// Determine whether the allow-reciprocal flag is set.
  bool hasAllowReciprocal() const {
    return cast<llvm::Instruction>(Val)->hasAllowReciprocal();
  }
  /// Set or clear the allow-reciprocal flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReciprocal(bool B);
  /// Determine whether the allow-contract flag is set.
  bool hasAllowContract() const {
    return cast<llvm::Instruction>(Val)->hasAllowContract();
  }
  /// Set or clear the allow-contract flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowContract(bool B);
  /// Determine whether the approximate-math-functions flag is set.
  bool hasApproxFunc() const {
    return cast<llvm::Instruction>(Val)->hasApproxFunc();
  }
  /// Set or clear the approximate-math-functions flag on this instruction,
  /// which must be an operator which supports this flag. See LangRef.html for
  /// the meaning of this flag.
  void setHasApproxFunc(bool B);
  /// Convenience function for getting all the fast-math flags, which must be an
  /// operator which supports these flags. See LangRef.html for the meaning of
  /// these flags.
  FastMathFlags getFastMathFlags() const {
    return cast<llvm::Instruction>(Val)->getFastMathFlags();
  }
  /// Convenience function for setting multiple fast-math flags on this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void setFastMathFlags(FastMathFlags FMF);
  /// Convenience function for transferring all fast-math flag values to this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void copyFastMathFlags(FastMathFlags FMF);

  bool isAssociative() const {
    return cast<llvm::Instruction>(Val)->isAssociative();
  }

  bool isCommutative() const {
    return cast<llvm::Instruction>(Val)->isCommutative();
  }

  bool isIdempotent() const {
    return cast<llvm::Instruction>(Val)->isIdempotent();
  }

  bool isNilpotent() const {
    return cast<llvm::Instruction>(Val)->isNilpotent();
  }

  bool mayWriteToMemory() const {
    return cast<llvm::Instruction>(Val)->mayWriteToMemory();
  }

  bool mayReadFromMemory() const {
    return cast<llvm::Instruction>(Val)->mayReadFromMemory();
  }
  bool mayReadOrWriteMemory() const {
    return cast<llvm::Instruction>(Val)->mayReadOrWriteMemory();
  }

  bool isAtomic() const { return cast<llvm::Instruction>(Val)->isAtomic(); }

  bool hasAtomicLoad() const {
    return cast<llvm::Instruction>(Val)->hasAtomicLoad();
  }

  bool hasAtomicStore() const {
    return cast<llvm::Instruction>(Val)->hasAtomicStore();
  }

  bool isVolatile() const { return cast<llvm::Instruction>(Val)->isVolatile(); }

  Type *getAccessType() const;

  bool mayThrow(bool IncludePhaseOneUnwind = false) const {
    return cast<llvm::Instruction>(Val)->mayThrow(IncludePhaseOneUnwind);
  }

  bool isFenceLike() const {
    return cast<llvm::Instruction>(Val)->isFenceLike();
  }

  bool mayHaveSideEffects() const {
    return cast<llvm::Instruction>(Val)->mayHaveSideEffects();
  }

  // TODO: Missing functions.

  bool isStackSaveOrRestoreIntrinsic() const {
    auto *I = cast<llvm::Instruction>(Val);
    return match(I,
                 PatternMatch::m_Intrinsic<llvm::Intrinsic::stackrestore>()) ||
           match(I, PatternMatch::m_Intrinsic<llvm::Intrinsic::stacksave>());
  }

  /// We consider \p I as a Memory Dependency Candidate instruction if it
  /// reads/write memory or if it has side-effects. This is used by the
  /// dependency graph.
  bool isMemDepCandidate() const {
    auto *I = cast<llvm::Instruction>(Val);
    return I->mayReadOrWriteMemory() &&
           (!isa<llvm::IntrinsicInst>(I) ||
            (cast<llvm::IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::sideeffect &&
             cast<llvm::IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::pseudoprobe));
  }

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS) const override;
#endif
};

/// Instructions that contain a single LLVM Instruction can inherit from this.
template <typename LLVMT> class SingleLLVMInstructionImpl : public Instruction {
  SingleLLVMInstructionImpl(ClassID ID, Opcode Opc, llvm::Instruction *I,
                            sandboxir::Context &SBCtx)
      : Instruction(ID, Opc, I, SBCtx) {}

  // All instructions are friends with this so they can call the constructor.
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"
  friend class UnaryInstruction;
  friend class CallBase;
  friend class FuncletPadInst;
  friend class CmpInst;

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
#ifndef NDEBUG
  void verify() const final { assert(isa<LLVMT>(Val) && "Expected LLVMT!"); }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class FenceInst : public SingleLLVMInstructionImpl<llvm::FenceInst> {
  FenceInst(llvm::FenceInst *FI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Fence, Opcode::Fence, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static FenceInst *create(AtomicOrdering Ordering, BBIterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           SyncScope::ID SSID = SyncScope::System);
  /// Returns the ordering constraint of this fence instruction.
  AtomicOrdering getOrdering() const {
    return cast<llvm::FenceInst>(Val)->getOrdering();
  }
  /// Sets the ordering constraint of this fence instruction.  May only be
  /// Acquire, Release, AcquireRelease, or SequentiallyConsistent.
  void setOrdering(AtomicOrdering Ordering);
  /// Returns the synchronization scope ID of this fence instruction.
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::FenceInst>(Val)->getSyncScopeID();
  }
  /// Sets the synchronization scope ID of this fence instruction.
  void setSyncScopeID(SyncScope::ID SSID);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Fence;
  }
};

class SelectInst : public SingleLLVMInstructionImpl<llvm::SelectInst> {
  /// Use Context::createSelectInst(). Don't call the
  /// constructor directly.
  SelectInst(llvm::SelectInst *CI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Select, Opcode::Select, CI, Ctx) {}
  friend Context; // for SelectInst()
  static Value *createCommon(Value *Cond, Value *True, Value *False,
                             const Twine &Name, IRBuilder<> &Builder,
                             Context &Ctx);

public:
  static Value *create(Value *Cond, Value *True, Value *False,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Cond, Value *True, Value *False,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");

  const Value *getCondition() const { return getOperand(0); }
  const Value *getTrueValue() const { return getOperand(1); }
  const Value *getFalseValue() const { return getOperand(2); }
  Value *getCondition() { return getOperand(0); }
  Value *getTrueValue() { return getOperand(1); }
  Value *getFalseValue() { return getOperand(2); }

  void setCondition(Value *New) { setOperand(0, New); }
  void setTrueValue(Value *New) { setOperand(1, New); }
  void setFalseValue(Value *New) { setOperand(2, New); }
  void swapValues();

  /// Return a string if the specified operands are invalid for a select
  /// operation, otherwise return null.
  static const char *areInvalidOperands(Value *Cond, Value *True,
                                        Value *False) {
    return llvm::SelectInst::areInvalidOperands(Cond->Val, True->Val,
                                                False->Val);
  }

  /// For isa/dyn_cast.
  static bool classof(const Value *From);
};

class InsertElementInst final
    : public SingleLLVMInstructionImpl<llvm::InsertElementInst> {
  /// Use Context::createInsertElementInst() instead.
  InsertElementInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::InsertElement, Opcode::InsertElement,
                                  I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()

public:
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::InsertElement;
  }
  static bool isValidOperands(const Value *Vec, const Value *NewElt,
                              const Value *Idx) {
    return llvm::InsertElementInst::isValidOperands(Vec->Val, NewElt->Val,
                                                    Idx->Val);
  }
};

class ExtractElementInst final
    : public SingleLLVMInstructionImpl<llvm::ExtractElementInst> {
  /// Use Context::createExtractElementInst() instead.
  ExtractElementInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::ExtractElement,
                                  Opcode::ExtractElement, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static Value *create(Value *Vec, Value *Idx, Instruction *InsertBefore,
                       Context &Ctx, const Twine &Name = "");
  static Value *create(Value *Vec, Value *Idx, BasicBlock *InsertAtEnd,
                       Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ExtractElement;
  }

  static bool isValidOperands(const Value *Vec, const Value *Idx) {
    return llvm::ExtractElementInst::isValidOperands(Vec->Val, Idx->Val);
  }
  Value *getVectorOperand() { return getOperand(0); }
  Value *getIndexOperand() { return getOperand(1); }
  const Value *getVectorOperand() const { return getOperand(0); }
  const Value *getIndexOperand() const { return getOperand(1); }
  VectorType *getVectorOperandType() const;
};

class ShuffleVectorInst final
    : public SingleLLVMInstructionImpl<llvm::ShuffleVectorInst> {
  /// Use Context::createShuffleVectorInst() instead.
  ShuffleVectorInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::ShuffleVector, Opcode::ShuffleVector,
                                  I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()

public:
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ShuffleVector;
  }

  /// Swap the operands and adjust the mask to preserve the semantics of the
  /// instruction.
  void commute();

  /// Return true if a shufflevector instruction can be formed with the
  /// specified operands.
  static bool isValidOperands(const Value *V1, const Value *V2,
                              const Value *Mask) {
    return llvm::ShuffleVectorInst::isValidOperands(V1->Val, V2->Val,
                                                    Mask->Val);
  }
  static bool isValidOperands(const Value *V1, const Value *V2,
                              ArrayRef<int> Mask) {
    return llvm::ShuffleVectorInst::isValidOperands(V1->Val, V2->Val, Mask);
  }

  /// Overload to return most specific vector type.
  VectorType *getType() const;

  /// Return the shuffle mask value of this instruction for the given element
  /// index. Return PoisonMaskElem if the element is undef.
  int getMaskValue(unsigned Elt) const {
    return cast<llvm::ShuffleVectorInst>(Val)->getMaskValue(Elt);
  }

  /// Convert the input shuffle mask operand to a vector of integers. Undefined
  /// elements of the mask are returned as PoisonMaskElem.
  static void getShuffleMask(const Constant *Mask,
                             SmallVectorImpl<int> &Result) {
    llvm::ShuffleVectorInst::getShuffleMask(cast<llvm::Constant>(Mask->Val),
                                            Result);
  }

  /// Return the mask for this instruction as a vector of integers. Undefined
  /// elements of the mask are returned as PoisonMaskElem.
  void getShuffleMask(SmallVectorImpl<int> &Result) const {
    cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask(Result);
  }

  /// Return the mask for this instruction, for use in bitcode.
  Constant *getShuffleMaskForBitcode() const;

  static Constant *convertShuffleMaskForBitcode(ArrayRef<int> Mask,
                                                Type *ResultTy);

  void setShuffleMask(ArrayRef<int> Mask);

  ArrayRef<int> getShuffleMask() const {
    return cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask();
  }

  /// Return true if this shuffle returns a vector with a different number of
  /// elements than its source vectors.
  /// Examples: shufflevector <4 x n> A, <4 x n> B, <1,2,3>
  ///           shufflevector <4 x n> A, <4 x n> B, <1,2,3,4,5>
  bool changesLength() const {
    return cast<llvm::ShuffleVectorInst>(Val)->changesLength();
  }

  /// Return true if this shuffle returns a vector with a greater number of
  /// elements than its source vectors.
  /// Example: shufflevector <2 x n> A, <2 x n> B, <1,2,3>
  bool increasesLength() const {
    return cast<llvm::ShuffleVectorInst>(Val)->increasesLength();
  }

  /// Return true if this shuffle mask chooses elements from exactly one source
  /// vector.
  /// Example: <7,5,undef,7>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isSingleSourceMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSingleSourceMask(Mask, NumSrcElts);
  }
  static bool isSingleSourceMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSingleSourceMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from exactly one source
  /// vector without changing the length of that vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <3,0,undef,3>
  bool isSingleSource() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSingleSource();
  }

  /// Return true if this shuffle mask chooses elements from exactly one source
  /// vector without lane crossings. A shuffle using this mask is not
  /// necessarily a no-op because it may change the number of elements from its
  /// input vectors or it may provide demanded bits knowledge via undef lanes.
  /// Example: <undef,undef,2,3>
  static bool isIdentityMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isIdentityMask(Mask, NumSrcElts);
  }
  static bool isIdentityMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isIdentityMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from exactly one source
  /// vector without lane crossings and does not change the number of elements
  /// from its input vectors.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <4,undef,6,undef>
  bool isIdentity() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentity();
  }

  /// Return true if this shuffle lengthens exactly one source vector with
  /// undefs in the high elements.
  bool isIdentityWithPadding() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentityWithPadding();
  }

  /// Return true if this shuffle extracts the first N elements of exactly one
  /// source vector.
  bool isIdentityWithExtract() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentityWithExtract();
  }

  /// Return true if this shuffle concatenates its 2 source vectors. This
  /// returns false if either input is undefined. In that case, the shuffle is
  /// is better classified as an identity with padding operation.
  bool isConcat() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isConcat();
  }

  /// Return true if this shuffle mask chooses elements from its source vectors
  /// without lane crossings. A shuffle using this mask would be
  /// equivalent to a vector select with a constant condition operand.
  /// Example: <4,1,6,undef>
  /// This returns false if the mask does not choose from both input vectors.
  /// In that case, the shuffle is better classified as an identity shuffle.
  /// This assumes that vector operands are the same length as the mask
  /// (a length-changing shuffle can never be equivalent to a vector select).
  static bool isSelectMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSelectMask(Mask, NumSrcElts);
  }
  static bool isSelectMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSelectMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from its source vectors
  /// without lane crossings and all operands have the same number of elements.
  /// In other words, this shuffle is equivalent to a vector select with a
  /// constant condition operand.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <undef,1,6,3>
  /// This returns false if the mask does not choose from both input vectors.
  /// In that case, the shuffle is better classified as an identity shuffle.
  bool isSelect() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSelect();
  }

  /// Return true if this shuffle mask swaps the order of elements from exactly
  /// one source vector.
  /// Example: <7,6,undef,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isReverseMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isReverseMask(Mask, NumSrcElts);
  }
  static bool isReverseMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isReverseMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle swaps the order of elements from exactly
  /// one source vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <3,undef,1,undef>
  bool isReverse() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isReverse();
  }

  /// Return true if this shuffle mask chooses all elements with the same value
  /// as the first element of exactly one source vector.
  /// Example: <4,undef,undef,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isZeroEltSplatMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isZeroEltSplatMask(Mask, NumSrcElts);
  }
  static bool isZeroEltSplatMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isZeroEltSplatMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if all elements of this shuffle are the same value as the
  /// first element of exactly one source vector without changing the length
  /// of that vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <undef,0,undef,0>
  bool isZeroEltSplat() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isZeroEltSplat();
  }

  /// Return true if this shuffle mask is a transpose mask.
  /// Transpose vector masks transpose a 2xn matrix. They read corresponding
  /// even- or odd-numbered vector elements from two n-dimensional source
  /// vectors and write each result into consecutive elements of an
  /// n-dimensional destination vector. Two shuffles are necessary to complete
  /// the transpose, one for the even elements and another for the odd elements.
  /// This description closely follows how the TRN1 and TRN2 AArch64
  /// instructions operate.
  ///
  /// For example, a simple 2x2 matrix can be transposed with:
  ///
  ///   ; Original matrix
  ///   m0 = < a, b >
  ///   m1 = < c, d >
  ///
  ///   ; Transposed matrix
  ///   t0 = < a, c > = shufflevector m0, m1, < 0, 2 >
  ///   t1 = < b, d > = shufflevector m0, m1, < 1, 3 >
  ///
  /// For matrices having greater than n columns, the resulting nx2 transposed
  /// matrix is stored in two result vectors such that one vector contains
  /// interleaved elements from all the even-numbered rows and the other vector
  /// contains interleaved elements from all the odd-numbered rows. For example,
  /// a 2x4 matrix can be transposed with:
  ///
  ///   ; Original matrix
  ///   m0 = < a, b, c, d >
  ///   m1 = < e, f, g, h >
  ///
  ///   ; Transposed matrix
  ///   t0 = < a, e, c, g > = shufflevector m0, m1 < 0, 4, 2, 6 >
  ///   t1 = < b, f, d, h > = shufflevector m0, m1 < 1, 5, 3, 7 >
  static bool isTransposeMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isTransposeMask(Mask, NumSrcElts);
  }
  static bool isTransposeMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isTransposeMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle transposes the elements of its inputs without
  /// changing the length of the vectors. This operation may also be known as a
  /// merge or interleave. See the description for isTransposeMask() for the
  /// exact specification.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <0,4,2,6>
  bool isTranspose() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isTranspose();
  }

  /// Return true if this shuffle mask is a splice mask, concatenating the two
  /// inputs together and then extracts an original width vector starting from
  /// the splice index.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <1,2,3,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isSpliceMask(ArrayRef<int> Mask, int NumSrcElts, int &Index) {
    return llvm::ShuffleVectorInst::isSpliceMask(Mask, NumSrcElts, Index);
  }
  static bool isSpliceMask(const Constant *Mask, int NumSrcElts, int &Index) {
    return llvm::ShuffleVectorInst::isSpliceMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, Index);
  }

  /// Return true if this shuffle splices two inputs without changing the length
  /// of the vectors. This operation concatenates the two inputs together and
  /// then extracts an original width vector starting from the splice index.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <1,2,3,4>
  bool isSplice(int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSplice(Index);
  }

  /// Return true if this shuffle mask is an extract subvector mask.
  /// A valid extract subvector mask returns a smaller vector from a single
  /// source operand. The base extraction index is returned as well.
  static bool isExtractSubvectorMask(ArrayRef<int> Mask, int NumSrcElts,
                                     int &Index) {
    return llvm::ShuffleVectorInst::isExtractSubvectorMask(Mask, NumSrcElts,
                                                           Index);
  }
  static bool isExtractSubvectorMask(const Constant *Mask, int NumSrcElts,
                                     int &Index) {
    return llvm::ShuffleVectorInst::isExtractSubvectorMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, Index);
  }

  /// Return true if this shuffle mask is an extract subvector mask.
  bool isExtractSubvectorMask(int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isExtractSubvectorMask(Index);
  }

  /// Return true if this shuffle mask is an insert subvector mask.
  /// A valid insert subvector mask inserts the lowest elements of a second
  /// source operand into an in-place first source operand.
  /// Both the sub vector width and the insertion index is returned.
  static bool isInsertSubvectorMask(ArrayRef<int> Mask, int NumSrcElts,
                                    int &NumSubElts, int &Index) {
    return llvm::ShuffleVectorInst::isInsertSubvectorMask(Mask, NumSrcElts,
                                                          NumSubElts, Index);
  }
  static bool isInsertSubvectorMask(const Constant *Mask, int NumSrcElts,
                                    int &NumSubElts, int &Index) {
    return llvm::ShuffleVectorInst::isInsertSubvectorMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, NumSubElts, Index);
  }

  /// Return true if this shuffle mask is an insert subvector mask.
  bool isInsertSubvectorMask(int &NumSubElts, int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isInsertSubvectorMask(NumSubElts,
                                                                     Index);
  }

  /// Return true if this shuffle mask replicates each of the \p VF elements
  /// in a vector \p ReplicationFactor times.
  /// For example, the mask for \p ReplicationFactor=3 and \p VF=4 is:
  ///   <0,0,0,1,1,1,2,2,2,3,3,3>
  static bool isReplicationMask(ArrayRef<int> Mask, int &ReplicationFactor,
                                int &VF) {
    return llvm::ShuffleVectorInst::isReplicationMask(Mask, ReplicationFactor,
                                                      VF);
  }
  static bool isReplicationMask(const Constant *Mask, int &ReplicationFactor,
                                int &VF) {
    return llvm::ShuffleVectorInst::isReplicationMask(
        cast<llvm::Constant>(Mask->Val), ReplicationFactor, VF);
  }

  /// Return true if this shuffle mask is a replication mask.
  bool isReplicationMask(int &ReplicationFactor, int &VF) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isReplicationMask(
        ReplicationFactor, VF);
  }

  /// Return true if this shuffle mask represents "clustered" mask of size VF,
  /// i.e. each index between [0..VF) is used exactly once in each submask of
  /// size VF.
  /// For example, the mask for \p VF=4 is:
  /// 0, 1, 2, 3, 3, 2, 0, 1 - "clustered", because each submask of size 4
  /// (0,1,2,3 and 3,2,0,1) uses indices [0..VF) exactly one time.
  /// 0, 1, 2, 3, 3, 3, 1, 0 - not "clustered", because
  ///                          element 3 is used twice in the second submask
  ///                          (3,3,1,0) and index 2 is not used at all.
  static bool isOneUseSingleSourceMask(ArrayRef<int> Mask, int VF) {
    return llvm::ShuffleVectorInst::isOneUseSingleSourceMask(Mask, VF);
  }

  /// Return true if this shuffle mask is a one-use-single-source("clustered")
  /// mask.
  bool isOneUseSingleSourceMask(int VF) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isOneUseSingleSourceMask(VF);
  }

  /// Change values in a shuffle permute mask assuming the two vector operands
  /// of length InVecNumElts have swapped position.
  static void commuteShuffleMask(MutableArrayRef<int> Mask,
                                 unsigned InVecNumElts) {
    llvm::ShuffleVectorInst::commuteShuffleMask(Mask, InVecNumElts);
  }

  /// Return if this shuffle interleaves its two input vectors together.
  bool isInterleave(unsigned Factor) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isInterleave(Factor);
  }

  /// Return true if the mask interleaves one or more input vectors together.
  ///
  /// I.e. <0, LaneLen, ... , LaneLen*(Factor - 1), 1, LaneLen + 1, ...>
  /// E.g. For a Factor of 2 (LaneLen=4):
  ///   <0, 4, 1, 5, 2, 6, 3, 7>
  /// E.g. For a Factor of 3 (LaneLen=4):
  ///   <4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3, 12>
  /// E.g. For a Factor of 4 (LaneLen=2):
  ///   <0, 2, 6, 4, 1, 3, 7, 5>
  ///
  /// NumInputElts is the total number of elements in the input vectors.
  ///
  /// StartIndexes are the first indexes of each vector being interleaved,
  /// substituting any indexes that were undef
  /// E.g. <4, -1, 2, 5, 1, 3> (Factor=3): StartIndexes=<4, 0, 2>
  ///
  /// Note that this does not check if the input vectors are consecutive:
  /// It will return true for masks such as
  /// <0, 4, 6, 1, 5, 7> (Factor=3, LaneLen=2)
  static bool isInterleaveMask(ArrayRef<int> Mask, unsigned Factor,
                               unsigned NumInputElts,
                               SmallVectorImpl<unsigned> &StartIndexes) {
    return llvm::ShuffleVectorInst::isInterleaveMask(Mask, Factor, NumInputElts,
                                                     StartIndexes);
  }
  static bool isInterleaveMask(ArrayRef<int> Mask, unsigned Factor,
                               unsigned NumInputElts) {
    return llvm::ShuffleVectorInst::isInterleaveMask(Mask, Factor,
                                                     NumInputElts);
  }

  /// Check if the mask is a DE-interleave mask of the given factor
  /// \p Factor like:
  ///     <Index, Index+Factor, ..., Index+(NumElts-1)*Factor>
  static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor,
                                         unsigned &Index) {
    return llvm::ShuffleVectorInst::isDeInterleaveMaskOfFactor(Mask, Factor,
                                                               Index);
  }
  static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor) {
    return llvm::ShuffleVectorInst::isDeInterleaveMaskOfFactor(Mask, Factor);
  }

  /// Checks if the shuffle is a bit rotation of the first operand across
  /// multiple subelements, e.g:
  ///
  /// shuffle <8 x i8> %a, <8 x i8> poison, <8 x i32> <1, 0, 3, 2, 5, 4, 7, 6>
  ///
  /// could be expressed as
  ///
  /// rotl <4 x i16> %a, 8
  ///
  /// If it can be expressed as a rotation, returns the number of subelements to
  /// group by in NumSubElts and the number of bits to rotate left in RotateAmt.
  static bool isBitRotateMask(ArrayRef<int> Mask, unsigned EltSizeInBits,
                              unsigned MinSubElts, unsigned MaxSubElts,
                              unsigned &NumSubElts, unsigned &RotateAmt) {
    return llvm::ShuffleVectorInst::isBitRotateMask(
        Mask, EltSizeInBits, MinSubElts, MaxSubElts, NumSubElts, RotateAmt);
  }
};

class InsertValueInst
    : public SingleLLVMInstructionImpl<llvm::InsertValueInst> {
  /// Use Context::createInsertValueInst(). Don't call the constructor directly.
  InsertValueInst(llvm::InsertValueInst *IVI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::InsertValue, Opcode::InsertValue,
                                  IVI, Ctx) {}
  friend Context; // for InsertValueInst()

public:
  static Value *create(Value *Agg, Value *Val, ArrayRef<unsigned> Idxs,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::InsertValue;
  }

  using idx_iterator = llvm::InsertValueInst::idx_iterator;
  inline idx_iterator idx_begin() const {
    return cast<llvm::InsertValueInst>(Val)->idx_begin();
  }
  inline idx_iterator idx_end() const {
    return cast<llvm::InsertValueInst>(Val)->idx_end();
  }
  inline iterator_range<idx_iterator> indices() const {
    return cast<llvm::InsertValueInst>(Val)->indices();
  }

  Value *getAggregateOperand() {
    return getOperand(getAggregateOperandIndex());
  }
  const Value *getAggregateOperand() const {
    return getOperand(getAggregateOperandIndex());
  }
  static unsigned getAggregateOperandIndex() {
    return llvm::InsertValueInst::getAggregateOperandIndex();
  }

  Value *getInsertedValueOperand() {
    return getOperand(getInsertedValueOperandIndex());
  }
  const Value *getInsertedValueOperand() const {
    return getOperand(getInsertedValueOperandIndex());
  }
  static unsigned getInsertedValueOperandIndex() {
    return llvm::InsertValueInst::getInsertedValueOperandIndex();
  }

  ArrayRef<unsigned> getIndices() const {
    return cast<llvm::InsertValueInst>(Val)->getIndices();
  }

  unsigned getNumIndices() const {
    return cast<llvm::InsertValueInst>(Val)->getNumIndices();
  }

  unsigned hasIndices() const {
    return cast<llvm::InsertValueInst>(Val)->hasIndices();
  }
};

class BranchInst : public SingleLLVMInstructionImpl<llvm::BranchInst> {
  /// Use Context::createBranchInst(). Don't call the constructor directly.
  BranchInst(llvm::BranchInst *BI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Br, Opcode::Br, BI, Ctx) {}
  friend Context; // for BranchInst()

public:
  static BranchInst *create(BasicBlock *IfTrue, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, BasicBlock *InsertAtEnd, Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  bool isUnconditional() const {
    return cast<llvm::BranchInst>(Val)->isUnconditional();
  }
  bool isConditional() const {
    return cast<llvm::BranchInst>(Val)->isConditional();
  }
  Value *getCondition() const;
  void setCondition(Value *V) { setOperand(0, V); }
  unsigned getNumSuccessors() const { return 1 + isConditional(); }
  BasicBlock *getSuccessor(unsigned SuccIdx) const;
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc);
  void swapSuccessors() { swapOperandsInternal(1, 2); }

private:
  struct LLVMBBToSBBB {
    Context &Ctx;
    LLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock *operator()(llvm::BasicBlock *BB) const;
  };

  struct ConstLLVMBBToSBBB {
    Context &Ctx;
    ConstLLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    const BasicBlock *operator()(const llvm::BasicBlock *BB) const;
  };

public:
  using sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::succ_op_iterator, LLVMBBToSBBB>;
  iterator_range<sb_succ_op_iterator> successors() {
    iterator_range<llvm::BranchInst::succ_op_iterator> LLVMRange =
        cast<llvm::BranchInst>(Val)->successors();
    LLVMBBToSBBB BBMap(Ctx);
    sb_succ_op_iterator MappedBegin = map_iterator(LLVMRange.begin(), BBMap);
    sb_succ_op_iterator MappedEnd = map_iterator(LLVMRange.end(), BBMap);
    return make_range(MappedBegin, MappedEnd);
  }

  using const_sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::const_succ_op_iterator,
                      ConstLLVMBBToSBBB>;
  iterator_range<const_sb_succ_op_iterator> successors() const {
    iterator_range<llvm::BranchInst::const_succ_op_iterator> ConstLLVMRange =
        static_cast<const llvm::BranchInst *>(cast<llvm::BranchInst>(Val))
            ->successors();
    ConstLLVMBBToSBBB ConstBBMap(Ctx);
    const_sb_succ_op_iterator ConstMappedBegin =
        map_iterator(ConstLLVMRange.begin(), ConstBBMap);
    const_sb_succ_op_iterator ConstMappedEnd =
        map_iterator(ConstLLVMRange.end(), ConstBBMap);
    return make_range(ConstMappedBegin, ConstMappedEnd);
  }
};

/// An abstract class, parent of unary instructions.
class UnaryInstruction
    : public SingleLLVMInstructionImpl<llvm::UnaryInstruction> {
protected:
  UnaryInstruction(ClassID ID, Opcode Opc, llvm::Instruction *LLVMI,
                   Context &Ctx)
      : SingleLLVMInstructionImpl(ID, Opc, LLVMI, Ctx) {}

public:
  static bool classof(const Instruction *I) {
    return isa<LoadInst>(I) || isa<CastInst>(I) || isa<FreezeInst>(I);
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

class ExtractValueInst : public UnaryInstruction {
  /// Use Context::createExtractValueInst() instead.
  ExtractValueInst(llvm::ExtractValueInst *EVI, Context &Ctx)
      : UnaryInstruction(ClassID::ExtractValue, Opcode::ExtractValue, EVI,
                         Ctx) {}
  friend Context; // for ExtractValueInst()

public:
  static Value *create(Value *Agg, ArrayRef<unsigned> Idxs, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ExtractValue;
  }

  /// Returns the type of the element that would be extracted
  /// with an extractvalue instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified type.
  static Type *getIndexedType(Type *Agg, ArrayRef<unsigned> Idxs);

  using idx_iterator = llvm::ExtractValueInst::idx_iterator;

  inline idx_iterator idx_begin() const {
    return cast<llvm::ExtractValueInst>(Val)->idx_begin();
  }
  inline idx_iterator idx_end() const {
    return cast<llvm::ExtractValueInst>(Val)->idx_end();
  }
  inline iterator_range<idx_iterator> indices() const {
    return cast<llvm::ExtractValueInst>(Val)->indices();
  }

  Value *getAggregateOperand() {
    return getOperand(getAggregateOperandIndex());
  }
  const Value *getAggregateOperand() const {
    return getOperand(getAggregateOperandIndex());
  }
  static unsigned getAggregateOperandIndex() {
    return llvm::ExtractValueInst::getAggregateOperandIndex();
  }

  ArrayRef<unsigned> getIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->getIndices();
  }

  unsigned getNumIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->getNumIndices();
  }

  unsigned hasIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->hasIndices();
  }
};

class VAArgInst : public UnaryInstruction {
  VAArgInst(llvm::VAArgInst *FI, Context &Ctx)
      : UnaryInstruction(ClassID::VAArg, Opcode::VAArg, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static VAArgInst *create(Value *List, Type *Ty, BBIterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           const Twine &Name = "");
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<VAArgInst *>(this)->getPointerOperand();
  }
  static unsigned getPointerOperandIndex() {
    return llvm::VAArgInst::getPointerOperandIndex();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::VAArg;
  }
};

class FreezeInst : public UnaryInstruction {
  FreezeInst(llvm::FreezeInst *FI, Context &Ctx)
      : UnaryInstruction(ClassID::Freeze, Opcode::Freeze, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static FreezeInst *create(Value *V, BBIterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Freeze;
  }
};

class LoadInst final : public UnaryInstruction {
  /// Use LoadInst::create() instead of calling the constructor.
  LoadInst(llvm::LoadInst *LI, Context &Ctx)
      : UnaryInstruction(ClassID::Load, Opcode::Load, LI, Ctx) {}
  friend Context; // for LoadInst()

public:
  /// Return true if this is a load from a volatile memory location.
  bool isVolatile() const { return cast<llvm::LoadInst>(Val)->isVolatile(); }
  /// Specify whether this is a volatile load or not.
  void setVolatile(bool V);

  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          Instruction *InsertBefore, Context &Ctx,
                          const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          Instruction *InsertBefore, bool IsVolatile,
                          Context &Ctx, const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          BasicBlock *InsertAtEnd, Context &Ctx,
                          const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          BasicBlock *InsertAtEnd, bool IsVolatile,
                          Context &Ctx, const Twine &Name = "");

  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<llvm::LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<llvm::LoadInst>(Val)->isSimple(); }
};

class StoreInst final : public SingleLLVMInstructionImpl<llvm::StoreInst> {
  /// Use StoreInst::create().
  StoreInst(llvm::StoreInst *SI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Store, Opcode::Store, SI, Ctx) {}
  friend Context; // for StoreInst()

public:
  /// Return true if this is a store from a volatile memory location.
  bool isVolatile() const { return cast<llvm::StoreInst>(Val)->isVolatile(); }
  /// Specify whether this is a volatile store or not.
  void setVolatile(bool V);

  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, bool IsVolatile,
                           Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, bool IsVolatile,
                           Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Value *getValueOperand() const;
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<llvm::StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<llvm::StoreInst>(Val)->isUnordered(); }
};

class UnreachableInst final : public Instruction {
  /// Use UnreachableInst::create() instead of calling the constructor.
  UnreachableInst(llvm::UnreachableInst *I, Context &Ctx)
      : Instruction(ClassID::Unreachable, Opcode::Unreachable, I, Ctx) {}
  friend Context;
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }

public:
  static UnreachableInst *create(Instruction *InsertBefore, Context &Ctx);
  static UnreachableInst *create(BasicBlock *InsertAtEnd, Context &Ctx);
  static bool classof(const Value *From);
  unsigned getNumSuccessors() const { return 0; }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("UnreachableInst has no operands!");
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
};

class ReturnInst final : public SingleLLVMInstructionImpl<llvm::ReturnInst> {
  /// Use ReturnInst::create() instead of calling the constructor.
  ReturnInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Ret, Opcode::Ret, I, Ctx) {}
  ReturnInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::Ret, I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()
  static ReturnInst *createCommon(Value *RetVal, IRBuilder<> &Builder,
                                  Context &Ctx);

public:
  static ReturnInst *create(Value *RetVal, Instruction *InsertBefore,
                            Context &Ctx);
  static ReturnInst *create(Value *RetVal, BasicBlock *InsertAtEnd,
                            Context &Ctx);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Ret;
  }
  /// \Returns null if there is no return value.
  Value *getReturnValue() const;
};

class CallBase : public SingleLLVMInstructionImpl<llvm::CallBase> {
  CallBase(ClassID ID, Opcode Opc, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ID, Opc, I, Ctx) {}
  friend class CallInst;   // For constructor.
  friend class InvokeInst; // For constructor.
  friend class CallBrInst; // For constructor.

public:
  static bool classof(const Value *From) {
    auto Opc = From->getSubclassID();
    return Opc == Instruction::ClassID::Call ||
           Opc == Instruction::ClassID::Invoke ||
           Opc == Instruction::ClassID::CallBr;
  }

  FunctionType *getFunctionType() const;

  op_iterator data_operands_begin() { return op_begin(); }
  const_op_iterator data_operands_begin() const {
    return const_cast<CallBase *>(this)->data_operands_begin();
  }
  op_iterator data_operands_end() {
    auto *LLVMCB = cast<llvm::CallBase>(Val);
    auto Dist = LLVMCB->data_operands_end() - LLVMCB->data_operands_begin();
    return op_begin() + Dist;
  }
  const_op_iterator data_operands_end() const {
    auto *LLVMCB = cast<llvm::CallBase>(Val);
    auto Dist = LLVMCB->data_operands_end() - LLVMCB->data_operands_begin();
    return op_begin() + Dist;
  }
  iterator_range<op_iterator> data_ops() {
    return make_range(data_operands_begin(), data_operands_end());
  }
  iterator_range<const_op_iterator> data_ops() const {
    return make_range(data_operands_begin(), data_operands_end());
  }
  bool data_operands_empty() const {
    return data_operands_end() == data_operands_begin();
  }
  unsigned data_operands_size() const {
    return std::distance(data_operands_begin(), data_operands_end());
  }
  bool isDataOperand(Use U) const {
    assert(this == U.getUser() &&
           "Only valid to query with a use of this instruction!");
    return cast<llvm::CallBase>(Val)->isDataOperand(U.LLVMUse);
  }
  unsigned getDataOperandNo(Use U) const {
    assert(isDataOperand(U) && "Data operand # out of range!");
    return cast<llvm::CallBase>(Val)->getDataOperandNo(U.LLVMUse);
  }

  /// Return the total number operands (not operand bundles) used by
  /// every operand bundle in this OperandBundleUser.
  unsigned getNumTotalBundleOperands() const {
    return cast<llvm::CallBase>(Val)->getNumTotalBundleOperands();
  }

  op_iterator arg_begin() { return op_begin(); }
  const_op_iterator arg_begin() const { return op_begin(); }
  op_iterator arg_end() {
    return data_operands_end() - getNumTotalBundleOperands();
  }
  const_op_iterator arg_end() const {
    return const_cast<CallBase *>(this)->arg_end();
  }
  iterator_range<op_iterator> args() {
    return make_range(arg_begin(), arg_end());
  }
  iterator_range<const_op_iterator> args() const {
    return make_range(arg_begin(), arg_end());
  }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return arg_end() - arg_begin(); }

  Value *getArgOperand(unsigned OpIdx) const {
    assert(OpIdx < arg_size() && "Out of bounds!");
    return getOperand(OpIdx);
  }
  void setArgOperand(unsigned OpIdx, Value *NewOp) {
    assert(OpIdx < arg_size() && "Out of bounds!");
    setOperand(OpIdx, NewOp);
  }

  Use getArgOperandUse(unsigned Idx) const {
    assert(Idx < arg_size() && "Out of bounds!");
    return getOperandUse(Idx);
  }
  Use getArgOperandUse(unsigned Idx) {
    assert(Idx < arg_size() && "Out of bounds!");
    return getOperandUse(Idx);
  }

  bool isArgOperand(Use U) const {
    return cast<llvm::CallBase>(Val)->isArgOperand(U.LLVMUse);
  }
  unsigned getArgOperandNo(Use U) const {
    return cast<llvm::CallBase>(Val)->getArgOperandNo(U.LLVMUse);
  }
  bool hasArgument(const Value *V) const { return is_contained(args(), V); }

  Value *getCalledOperand() const;
  Use getCalledOperandUse() const;

  Function *getCalledFunction() const;
  bool isIndirectCall() const {
    return cast<llvm::CallBase>(Val)->isIndirectCall();
  }
  bool isCallee(Use U) const {
    return cast<llvm::CallBase>(Val)->isCallee(U.LLVMUse);
  }
  Function *getCaller();
  const Function *getCaller() const {
    return const_cast<CallBase *>(this)->getCaller();
  }
  bool isMustTailCall() const {
    return cast<llvm::CallBase>(Val)->isMustTailCall();
  }
  bool isTailCall() const { return cast<llvm::CallBase>(Val)->isTailCall(); }
  Intrinsic::ID getIntrinsicID() const {
    return cast<llvm::CallBase>(Val)->getIntrinsicID();
  }
  void setCalledOperand(Value *V) { getCalledOperandUse().set(V); }
  void setCalledFunction(Function *F);
  CallingConv::ID getCallingConv() const {
    return cast<llvm::CallBase>(Val)->getCallingConv();
  }
  bool isInlineAsm() const { return cast<llvm::CallBase>(Val)->isInlineAsm(); }
};

class CallInst final : public CallBase {
  /// Use Context::createCallInst(). Don't call the
  /// constructor directly.
  CallInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::Call, Opcode::Call, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BBIterator WhereIt,
                          BasicBlock *WhereBB, Context &Ctx,
                          const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, Instruction *InsertBefore,
                          Context &Ctx, const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                          Context &Ctx, const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Call;
  }
};

class InvokeInst final : public CallBase {
  /// Use Context::createInvokeInst(). Don't call the
  /// constructor directly.
  InvokeInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::Invoke, Opcode::Invoke, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            const Twine &NameStr = "");
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, Instruction *InsertBefore,
                            Context &Ctx, const Twine &NameStr = "");
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                            Context &Ctx, const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Invoke;
  }
  BasicBlock *getNormalDest() const;
  BasicBlock *getUnwindDest() const;
  void setNormalDest(BasicBlock *BB);
  void setUnwindDest(BasicBlock *BB);
  LandingPadInst *getLandingPadInst() const;
  BasicBlock *getSuccessor(unsigned SuccIdx) const;
  void setSuccessor(unsigned SuccIdx, BasicBlock *NewSucc) {
    assert(SuccIdx < 2 && "Successor # out of range for invoke!");
    if (SuccIdx == 0)
      setNormalDest(NewSucc);
    else
      setUnwindDest(NewSucc);
  }
  unsigned getNumSuccessors() const {
    return cast<llvm::InvokeInst>(Val)->getNumSuccessors();
  }
};

class CallBrInst final : public CallBase {
  /// Use Context::createCallBrInst(). Don't call the
  /// constructor directly.
  CallBrInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::CallBr, Opcode::CallBr, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            const Twine &NameStr = "");
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, Instruction *InsertBefore,
                            Context &Ctx, const Twine &NameStr = "");
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                            Context &Ctx, const Twine &NameStr = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CallBr;
  }
  unsigned getNumIndirectDests() const {
    return cast<llvm::CallBrInst>(Val)->getNumIndirectDests();
  }
  Value *getIndirectDestLabel(unsigned Idx) const;
  Value *getIndirectDestLabelUse(unsigned Idx) const;
  BasicBlock *getDefaultDest() const;
  BasicBlock *getIndirectDest(unsigned Idx) const;
  SmallVector<BasicBlock *, 16> getIndirectDests() const;
  void setDefaultDest(BasicBlock *BB);
  void setIndirectDest(unsigned Idx, BasicBlock *BB);
  BasicBlock *getSuccessor(unsigned Idx) const;
  unsigned getNumSuccessors() const {
    return cast<llvm::CallBrInst>(Val)->getNumSuccessors();
  }
};

class LandingPadInst : public SingleLLVMInstructionImpl<llvm::LandingPadInst> {
  LandingPadInst(llvm::LandingPadInst *LP, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::LandingPad, Opcode::LandingPad, LP,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static LandingPadInst *create(Type *RetTy, unsigned NumReservedClauses,
                                BBIterator WhereIt, BasicBlock *WhereBB,
                                Context &Ctx, const Twine &Name = "");
  /// Return 'true' if this landingpad instruction is a
  /// cleanup. I.e., it should be run when unwinding even if its landing pad
  /// doesn't catch the exception.
  bool isCleanup() const {
    return cast<llvm::LandingPadInst>(Val)->isCleanup();
  }
  /// Indicate that this landingpad instruction is a cleanup.
  void setCleanup(bool V);

  // TODO: We are not implementing addClause() because we have no way to revert
  // it for now.

  /// Get the value of the clause at index Idx. Use isCatch/isFilter to
  /// determine what type of clause this is.
  Constant *getClause(unsigned Idx) const;

  /// Return 'true' if the clause and index Idx is a catch clause.
  bool isCatch(unsigned Idx) const {
    return cast<llvm::LandingPadInst>(Val)->isCatch(Idx);
  }
  /// Return 'true' if the clause and index Idx is a filter clause.
  bool isFilter(unsigned Idx) const {
    return cast<llvm::LandingPadInst>(Val)->isFilter(Idx);
  }
  /// Get the number of clauses for this landing pad.
  unsigned getNumClauses() const {
    return cast<llvm::LandingPadInst>(Val)->getNumOperands();
  }
  // TODO: We are not implementing reserveClauses() because we can't revert it.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::LandingPad;
  }
};

class FuncletPadInst : public SingleLLVMInstructionImpl<llvm::FuncletPadInst> {
  FuncletPadInst(ClassID SubclassID, Opcode Opc, llvm::Instruction *I,
                 Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opc, I, Ctx) {}
  friend class CatchPadInst;   // For constructor.
  friend class CleanupPadInst; // For constructor.

public:
  /// Return the number of funcletpad arguments.
  unsigned arg_size() const {
    return cast<llvm::FuncletPadInst>(Val)->arg_size();
  }
  /// Return the outer EH-pad this funclet is nested within.
  ///
  /// Note: This returns the associated CatchSwitchInst if this FuncletPadInst
  /// is a CatchPadInst.
  Value *getParentPad() const;
  void setParentPad(Value *ParentPad);
  /// Return the Idx-th funcletpad argument.
  Value *getArgOperand(unsigned Idx) const;
  /// Set the Idx-th funcletpad argument.
  void setArgOperand(unsigned Idx, Value *V);

  // TODO: Implement missing functions: arg_operands().
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchPad ||
           From->getSubclassID() == ClassID::CleanupPad;
  }
};

class CatchPadInst : public FuncletPadInst {
  CatchPadInst(llvm::CatchPadInst *CPI, Context &Ctx)
      : FuncletPadInst(ClassID::CatchPad, Opcode::CatchPad, CPI, Ctx) {}
  friend class Context; // For constructor.

public:
  CatchSwitchInst *getCatchSwitch() const;
  // TODO: We have not implemented setCatchSwitch() because we can't revert it
  // for now, as there is no CatchPadInst member function that can undo it.

  static CatchPadInst *create(Value *ParentPad, ArrayRef<Value *> Args,
                              BBIterator WhereIt, BasicBlock *WhereBB,
                              Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchPad;
  }
};

class CleanupPadInst : public FuncletPadInst {
  CleanupPadInst(llvm::CleanupPadInst *CPI, Context &Ctx)
      : FuncletPadInst(ClassID::CleanupPad, Opcode::CleanupPad, CPI, Ctx) {}
  friend class Context; // For constructor.

public:
  static CleanupPadInst *create(Value *ParentPad, ArrayRef<Value *> Args,
                                BBIterator WhereIt, BasicBlock *WhereBB,
                                Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CleanupPad;
  }
};

class CatchReturnInst
    : public SingleLLVMInstructionImpl<llvm::CatchReturnInst> {
  CatchReturnInst(llvm::CatchReturnInst *CRI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CatchRet, Opcode::CatchRet, CRI,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static CatchReturnInst *create(CatchPadInst *CatchPad, BasicBlock *BB,
                                 BBIterator WhereIt, BasicBlock *WhereBB,
                                 Context &Ctx);
  CatchPadInst *getCatchPad() const;
  void setCatchPad(CatchPadInst *CatchPad);
  BasicBlock *getSuccessor() const;
  void setSuccessor(BasicBlock *NewSucc);
  unsigned getNumSuccessors() {
    return cast<llvm::CatchReturnInst>(Val)->getNumSuccessors();
  }
  Value *getCatchSwitchParentPad() const;
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchRet;
  }
};

class CleanupReturnInst
    : public SingleLLVMInstructionImpl<llvm::CleanupReturnInst> {
  CleanupReturnInst(llvm::CleanupReturnInst *CRI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CleanupRet, Opcode::CleanupRet, CRI,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static CleanupReturnInst *create(CleanupPadInst *CleanupPad,
                                   BasicBlock *UnwindBB, BBIterator WhereIt,
                                   BasicBlock *WhereBB, Context &Ctx);
  bool hasUnwindDest() const {
    return cast<llvm::CleanupReturnInst>(Val)->hasUnwindDest();
  }
  bool unwindsToCaller() const {
    return cast<llvm::CleanupReturnInst>(Val)->unwindsToCaller();
  }
  CleanupPadInst *getCleanupPad() const;
  void setCleanupPad(CleanupPadInst *CleanupPad);
  unsigned getNumSuccessors() const {
    return cast<llvm::CleanupReturnInst>(Val)->getNumSuccessors();
  }
  BasicBlock *getUnwindDest() const;
  void setUnwindDest(BasicBlock *NewDest);

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CleanupRet;
  }
};

class GetElementPtrInst final
    : public SingleLLVMInstructionImpl<llvm::GetElementPtrInst> {
  /// Use Context::createGetElementPtrInst(). Don't call
  /// the constructor directly.
  GetElementPtrInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::GetElementPtr, Opcode::GetElementPtr,
                                  I, Ctx) {}
  GetElementPtrInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::GetElementPtr, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::GetElementPtr;
  }

  Type *getSourceElementType() const;
  Type *getResultElementType() const;
  unsigned getAddressSpace() const {
    return cast<llvm::GetElementPtrInst>(Val)->getAddressSpace();
  }

  inline op_iterator idx_begin() { return op_begin() + 1; }
  inline const_op_iterator idx_begin() const {
    return const_cast<GetElementPtrInst *>(this)->idx_begin();
  }
  inline op_iterator idx_end() { return op_end(); }
  inline const_op_iterator idx_end() const {
    return const_cast<GetElementPtrInst *>(this)->idx_end();
  }
  inline iterator_range<op_iterator> indices() {
    return make_range(idx_begin(), idx_end());
  }
  inline iterator_range<const_op_iterator> indices() const {
    return const_cast<GetElementPtrInst *>(this)->indices();
  }

  Value *getPointerOperand() const;
  static unsigned getPointerOperandIndex() {
    return llvm::GetElementPtrInst::getPointerOperandIndex();
  }
  Type *getPointerOperandType() const;
  unsigned getPointerAddressSpace() const {
    return cast<llvm::GetElementPtrInst>(Val)->getPointerAddressSpace();
  }
  unsigned getNumIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->getNumIndices();
  }
  bool hasIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasIndices();
  }
  bool hasAllConstantIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasAllConstantIndices();
  }
  GEPNoWrapFlags getNoWrapFlags() const {
    return cast<llvm::GetElementPtrInst>(Val)->getNoWrapFlags();
  }
  bool isInBounds() const {
    return cast<llvm::GetElementPtrInst>(Val)->isInBounds();
  }
  bool hasNoUnsignedSignedWrap() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasNoUnsignedSignedWrap();
  }
  bool hasNoUnsignedWrap() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasNoUnsignedWrap();
  }
  bool accumulateConstantOffset(const DataLayout &DL, APInt &Offset) const {
    return cast<llvm::GetElementPtrInst>(Val)->accumulateConstantOffset(DL,
                                                                        Offset);
  }
  // TODO: Add missing member functions.
};

class CatchSwitchInst
    : public SingleLLVMInstructionImpl<llvm::CatchSwitchInst> {
public:
  CatchSwitchInst(llvm::CatchSwitchInst *CSI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CatchSwitch, Opcode::CatchSwitch,
                                  CSI, Ctx) {}

  static CatchSwitchInst *create(Value *ParentPad, BasicBlock *UnwindBB,
                                 unsigned NumHandlers, BBIterator WhereIt,
                                 BasicBlock *WhereBB, Context &Ctx,
                                 const Twine &Name = "");

  Value *getParentPad() const;
  void setParentPad(Value *ParentPad);

  bool hasUnwindDest() const {
    return cast<llvm::CatchSwitchInst>(Val)->hasUnwindDest();
  }
  bool unwindsToCaller() const {
    return cast<llvm::CatchSwitchInst>(Val)->unwindsToCaller();
  }
  BasicBlock *getUnwindDest() const;
  void setUnwindDest(BasicBlock *UnwindDest);

  unsigned getNumHandlers() const {
    return cast<llvm::CatchSwitchInst>(Val)->getNumHandlers();
  }

private:
  static BasicBlock *handler_helper(Value *V) { return cast<BasicBlock>(V); }
  static const BasicBlock *handler_helper(const Value *V) {
    return cast<BasicBlock>(V);
  }

public:
  using DerefFnTy = BasicBlock *(*)(Value *);
  using handler_iterator = mapped_iterator<op_iterator, DerefFnTy>;
  using handler_range = iterator_range<handler_iterator>;
  using ConstDerefFnTy = const BasicBlock *(*)(const Value *);
  using const_handler_iterator =
      mapped_iterator<const_op_iterator, ConstDerefFnTy>;
  using const_handler_range = iterator_range<const_handler_iterator>;

  handler_iterator handler_begin() {
    op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return handler_iterator(It, DerefFnTy(handler_helper));
  }
  const_handler_iterator handler_begin() const {
    const_op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return const_handler_iterator(It, ConstDerefFnTy(handler_helper));
  }
  handler_iterator handler_end() {
    return handler_iterator(op_end(), DerefFnTy(handler_helper));
  }
  const_handler_iterator handler_end() const {
    return const_handler_iterator(op_end(), ConstDerefFnTy(handler_helper));
  }
  handler_range handlers() {
    return make_range(handler_begin(), handler_end());
  }
  const_handler_range handlers() const {
    return make_range(handler_begin(), handler_end());
  }

  void addHandler(BasicBlock *Dest);

  // TODO: removeHandler() cannot be reverted because there is no equivalent
  // addHandler() with a handler_iterator to specify the position. So we can't
  // implement it for now.

  unsigned getNumSuccessors() const { return getNumOperands() - 1; }
  BasicBlock *getSuccessor(unsigned Idx) const {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    return cast<BasicBlock>(getOperand(Idx + 1));
  }
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    setOperand(Idx + 1, NewSucc);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchSwitch;
  }
};

class ResumeInst : public SingleLLVMInstructionImpl<llvm::ResumeInst> {
public:
  ResumeInst(llvm::ResumeInst *CSI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Resume, Opcode::Resume, CSI, Ctx) {}

  static ResumeInst *create(Value *Exn, BBIterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx);
  Value *getValue() const;
  unsigned getNumSuccessors() const {
    return cast<llvm::ResumeInst>(Val)->getNumSuccessors();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Resume;
  }
};

class SwitchInst : public SingleLLVMInstructionImpl<llvm::SwitchInst> {
public:
  SwitchInst(llvm::SwitchInst *SI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Switch, Opcode::Switch, SI, Ctx) {}

  static constexpr const unsigned DefaultPseudoIndex =
      llvm::SwitchInst::DefaultPseudoIndex;

  static SwitchInst *create(Value *V, BasicBlock *Dest, unsigned NumCases,
                            BasicBlock::iterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx, const Twine &Name = "");

  Value *getCondition() const;
  void setCondition(Value *V);
  BasicBlock *getDefaultDest() const;
  bool defaultDestUndefined() const {
    return cast<llvm::SwitchInst>(Val)->defaultDestUndefined();
  }
  void setDefaultDest(BasicBlock *DefaultCase);
  unsigned getNumCases() const {
    return cast<llvm::SwitchInst>(Val)->getNumCases();
  }

  using CaseHandle =
      llvm::SwitchInst::CaseHandleImpl<SwitchInst, ConstantInt, BasicBlock>;
  using ConstCaseHandle =
      llvm::SwitchInst::CaseHandleImpl<const SwitchInst, const ConstantInt,
                                       const BasicBlock>;
  using CaseIt = llvm::SwitchInst::CaseIteratorImpl<CaseHandle>;
  using ConstCaseIt = llvm::SwitchInst::CaseIteratorImpl<ConstCaseHandle>;

  /// Returns a read/write iterator that points to the first case in the
  /// SwitchInst.
  CaseIt case_begin() { return CaseIt(this, 0); }
  ConstCaseIt case_begin() const { return ConstCaseIt(this, 0); }
  /// Returns a read/write iterator that points one past the last in the
  /// SwitchInst.
  CaseIt case_end() { return CaseIt(this, getNumCases()); }
  ConstCaseIt case_end() const { return ConstCaseIt(this, getNumCases()); }
  /// Iteration adapter for range-for loops.
  iterator_range<CaseIt> cases() {
    return make_range(case_begin(), case_end());
  }
  iterator_range<ConstCaseIt> cases() const {
    return make_range(case_begin(), case_end());
  }
  CaseIt case_default() { return CaseIt(this, DefaultPseudoIndex); }
  ConstCaseIt case_default() const {
    return ConstCaseIt(this, DefaultPseudoIndex);
  }
  CaseIt findCaseValue(const ConstantInt *C) {
    return CaseIt(
        this,
        const_cast<const SwitchInst *>(this)->findCaseValue(C)->getCaseIndex());
  }
  ConstCaseIt findCaseValue(const ConstantInt *C) const {
    ConstCaseIt I = llvm::find_if(cases(), [C](const ConstCaseHandle &Case) {
      return Case.getCaseValue() == C;
    });
    if (I != case_end())
      return I;
    return case_default();
  }
  ConstantInt *findCaseDest(BasicBlock *BB);

  void addCase(ConstantInt *OnVal, BasicBlock *Dest);
  /// This method removes the specified case and its successor from the switch
  /// instruction. Note that this operation may reorder the remaining cases at
  /// index idx and above.
  /// Note:
  /// This action invalidates iterators for all cases following the one removed,
  /// including the case_end() iterator. It returns an iterator for the next
  /// case.
  CaseIt removeCase(CaseIt It);

  unsigned getNumSuccessors() const {
    return cast<llvm::SwitchInst>(Val)->getNumSuccessors();
  }
  BasicBlock *getSuccessor(unsigned Idx) const;
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Switch;
  }
};

class UnaryOperator : public UnaryInstruction {
  static Opcode getUnaryOpcode(llvm::Instruction::UnaryOps UnOp) {
    switch (UnOp) {
    case llvm::Instruction::FNeg:
      return Opcode::FNeg;
    case llvm::Instruction::UnaryOpsEnd:
      llvm_unreachable("Bad UnOp!");
    }
    llvm_unreachable("Unhandled UnOp!");
  }
  UnaryOperator(llvm::UnaryOperator *UO, Context &Ctx)
      : UnaryInstruction(ClassID::UnOp, getUnaryOpcode(UO->getOpcode()), UO,
                         Ctx) {}
  friend Context; // for constructor.
public:
  static Value *create(Instruction::Opcode Op, Value *OpV, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *OpV,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *OpV,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom, BBIterator WhereIt,
                                      BasicBlock *WhereBB, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom, BasicBlock *InsertAtEnd,
                                      Context &Ctx, const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::UnOp;
  }
};

class BinaryOperator : public SingleLLVMInstructionImpl<llvm::BinaryOperator> {
protected:
  static Opcode getBinOpOpcode(llvm::Instruction::BinaryOps BinOp) {
    switch (BinOp) {
    case llvm::Instruction::Add:
      return Opcode::Add;
    case llvm::Instruction::FAdd:
      return Opcode::FAdd;
    case llvm::Instruction::Sub:
      return Opcode::Sub;
    case llvm::Instruction::FSub:
      return Opcode::FSub;
    case llvm::Instruction::Mul:
      return Opcode::Mul;
    case llvm::Instruction::FMul:
      return Opcode::FMul;
    case llvm::Instruction::UDiv:
      return Opcode::UDiv;
    case llvm::Instruction::SDiv:
      return Opcode::SDiv;
    case llvm::Instruction::FDiv:
      return Opcode::FDiv;
    case llvm::Instruction::URem:
      return Opcode::URem;
    case llvm::Instruction::SRem:
      return Opcode::SRem;
    case llvm::Instruction::FRem:
      return Opcode::FRem;
    case llvm::Instruction::Shl:
      return Opcode::Shl;
    case llvm::Instruction::LShr:
      return Opcode::LShr;
    case llvm::Instruction::AShr:
      return Opcode::AShr;
    case llvm::Instruction::And:
      return Opcode::And;
    case llvm::Instruction::Or:
      return Opcode::Or;
    case llvm::Instruction::Xor:
      return Opcode::Xor;
    case llvm::Instruction::BinaryOpsEnd:
      llvm_unreachable("Bad BinOp!");
    }
    llvm_unreachable("Unhandled BinOp!");
  }
  BinaryOperator(llvm::BinaryOperator *BinOp, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::BinaryOperator,
                                  getBinOpOpcode(BinOp->getOpcode()), BinOp,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");

  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      BBIterator WhereIt, BasicBlock *WhereBB,
                                      Context &Ctx, const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      BasicBlock *InsertAtEnd, Context &Ctx,
                                      const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::BinaryOperator;
  }
  void swapOperands() { swapOperandsInternal(0, 1); }
};

/// An or instruction, which can be marked as "disjoint", indicating that the
/// inputs don't have a 1 in the same bit position. Meaning this instruction
/// can also be treated as an add.
class PossiblyDisjointInst : public BinaryOperator {
public:
  void setIsDisjoint(bool B);
  bool isDisjoint() const {
    return cast<llvm::PossiblyDisjointInst>(Val)->isDisjoint();
  }
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return isa<Instruction>(From) &&
           cast<Instruction>(From)->getOpcode() == Opcode::Or;
  }
};

class AtomicRMWInst : public SingleLLVMInstructionImpl<llvm::AtomicRMWInst> {
  AtomicRMWInst(llvm::AtomicRMWInst *Atomic, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::AtomicRMW,
                                  Instruction::Opcode::AtomicRMW, Atomic, Ctx) {
  }
  friend class Context; // For constructor.

public:
  using BinOp = llvm::AtomicRMWInst::BinOp;
  BinOp getOperation() const {
    return cast<llvm::AtomicRMWInst>(Val)->getOperation();
  }
  static StringRef getOperationName(BinOp Op) {
    return llvm::AtomicRMWInst::getOperationName(Op);
  }
  static bool isFPOperation(BinOp Op) {
    return llvm::AtomicRMWInst::isFPOperation(Op);
  }
  void setOperation(BinOp Op) {
    cast<llvm::AtomicRMWInst>(Val)->setOperation(Op);
  }
  Align getAlign() const { return cast<llvm::AtomicRMWInst>(Val)->getAlign(); }
  void setAlignment(Align Align);
  bool isVolatile() const {
    return cast<llvm::AtomicRMWInst>(Val)->isVolatile();
  }
  void setVolatile(bool V);
  AtomicOrdering getOrdering() const {
    return cast<llvm::AtomicRMWInst>(Val)->getOrdering();
  }
  void setOrdering(AtomicOrdering Ordering);
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::AtomicRMWInst>(Val)->getSyncScopeID();
  }
  void setSyncScopeID(SyncScope::ID SSID);
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<AtomicRMWInst *>(this)->getPointerOperand();
  }
  Value *getValOperand();
  const Value *getValOperand() const {
    return const_cast<AtomicRMWInst *>(this)->getValOperand();
  }
  unsigned getPointerAddressSpace() const {
    return cast<llvm::AtomicRMWInst>(Val)->getPointerAddressSpace();
  }
  bool isFloatingPointOperation() const {
    return cast<llvm::AtomicRMWInst>(Val)->isFloatingPointOperation();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::AtomicRMW;
  }

  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               BBIterator WhereIt, BasicBlock *WhereBB,
                               Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               Instruction *InsertBefore, Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               BasicBlock *InsertAtEnd, Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
};

class AtomicCmpXchgInst
    : public SingleLLVMInstructionImpl<llvm::AtomicCmpXchgInst> {
  AtomicCmpXchgInst(llvm::AtomicCmpXchgInst *Atomic, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::AtomicCmpXchg,
                                  Instruction::Opcode::AtomicCmpXchg, Atomic,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return the alignment of the memory that is being allocated by the
  /// instruction.
  Align getAlign() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getAlign();
  }

  void setAlignment(Align Align);
  /// Return true if this is a cmpxchg from a volatile memory
  /// location.
  bool isVolatile() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->isVolatile();
  }
  /// Specify whether this is a volatile cmpxchg.
  void setVolatile(bool V);
  /// Return true if this cmpxchg may spuriously fail.
  bool isWeak() const { return cast<llvm::AtomicCmpXchgInst>(Val)->isWeak(); }
  void setWeak(bool IsWeak);
  static bool isValidSuccessOrdering(AtomicOrdering Ordering) {
    return llvm::AtomicCmpXchgInst::isValidSuccessOrdering(Ordering);
  }
  static bool isValidFailureOrdering(AtomicOrdering Ordering) {
    return llvm::AtomicCmpXchgInst::isValidFailureOrdering(Ordering);
  }
  AtomicOrdering getSuccessOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getSuccessOrdering();
  }
  void setSuccessOrdering(AtomicOrdering Ordering);

  AtomicOrdering getFailureOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getFailureOrdering();
  }
  void setFailureOrdering(AtomicOrdering Ordering);
  AtomicOrdering getMergedOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getMergedOrdering();
  }
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getSyncScopeID();
  }
  void setSyncScopeID(SyncScope::ID SSID);
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getPointerOperand();
  }

  Value *getCompareOperand();
  const Value *getCompareOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getCompareOperand();
  }

  Value *getNewValOperand();
  const Value *getNewValOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getNewValOperand();
  }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getPointerAddressSpace();
  }

  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");
  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         Instruction *InsertBefore, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");
  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         BasicBlock *InsertAtEnd, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::AtomicCmpXchg;
  }
};

class AllocaInst final : public UnaryInstruction {
  AllocaInst(llvm::AllocaInst *AI, Context &Ctx)
      : UnaryInstruction(ClassID::Alloca, Instruction::Opcode::Alloca, AI,
                         Ctx) {}
  friend class Context; // For constructor.

public:
  static AllocaInst *create(Type *Ty, unsigned AddrSpace, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");
  static AllocaInst *create(Type *Ty, unsigned AddrSpace,
                            Instruction *InsertBefore, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");
  static AllocaInst *create(Type *Ty, unsigned AddrSpace,
                            BasicBlock *InsertAtEnd, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");

  /// Return true if there is an allocation size parameter to the allocation
  /// instruction that is not 1.
  bool isArrayAllocation() const {
    return cast<llvm::AllocaInst>(Val)->isArrayAllocation();
  }
  /// Get the number of elements allocated. For a simple allocation of a single
  /// element, this will return a constant 1 value.
  Value *getArraySize();
  const Value *getArraySize() const {
    return const_cast<AllocaInst *>(this)->getArraySize();
  }
  /// Overload to return most specific pointer type.
  PointerType *getType() const;
  /// Return the address space for the allocation.
  unsigned getAddressSpace() const {
    return cast<llvm::AllocaInst>(Val)->getAddressSpace();
  }
  /// Get allocation size in bytes. Returns std::nullopt if size can't be
  /// determined, e.g. in case of a VLA.
  std::optional<TypeSize> getAllocationSize(const DataLayout &DL) const {
    return cast<llvm::AllocaInst>(Val)->getAllocationSize(DL);
  }
  /// Get allocation size in bits. Returns std::nullopt if size can't be
  /// determined, e.g. in case of a VLA.
  std::optional<TypeSize> getAllocationSizeInBits(const DataLayout &DL) const {
    return cast<llvm::AllocaInst>(Val)->getAllocationSizeInBits(DL);
  }
  /// Return the type that is being allocated by the instruction.
  Type *getAllocatedType() const;
  /// for use only in special circumstances that need to generically
  /// transform a whole instruction (eg: IR linking and vectorization).
  void setAllocatedType(Type *Ty);
  /// Return the alignment of the memory that is being allocated by the
  /// instruction.
  Align getAlign() const { return cast<llvm::AllocaInst>(Val)->getAlign(); }
  void setAlignment(Align Align);
  /// Return true if this alloca is in the entry block of the function and is a
  /// constant size. If so, the code generator will fold it into the
  /// prolog/epilog code, so it is basically free.
  bool isStaticAlloca() const {
    return cast<llvm::AllocaInst>(Val)->isStaticAlloca();
  }
  /// Return true if this alloca is used as an inalloca argument to a call. Such
  /// allocas are never considered static even if they are in the entry block.
  bool isUsedWithInAlloca() const {
    return cast<llvm::AllocaInst>(Val)->isUsedWithInAlloca();
  }
  /// Specify whether this alloca is used to represent the arguments to a call.
  void setUsedWithInAlloca(bool V);

  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From))
      return I->getSubclassID() == Instruction::ClassID::Alloca;
    return false;
  }
};

class CastInst : public UnaryInstruction {
  static Opcode getCastOpcode(llvm::Instruction::CastOps CastOp) {
    switch (CastOp) {
    case llvm::Instruction::ZExt:
      return Opcode::ZExt;
    case llvm::Instruction::SExt:
      return Opcode::SExt;
    case llvm::Instruction::FPToUI:
      return Opcode::FPToUI;
    case llvm::Instruction::FPToSI:
      return Opcode::FPToSI;
    case llvm::Instruction::FPExt:
      return Opcode::FPExt;
    case llvm::Instruction::PtrToInt:
      return Opcode::PtrToInt;
    case llvm::Instruction::IntToPtr:
      return Opcode::IntToPtr;
    case llvm::Instruction::SIToFP:
      return Opcode::SIToFP;
    case llvm::Instruction::UIToFP:
      return Opcode::UIToFP;
    case llvm::Instruction::Trunc:
      return Opcode::Trunc;
    case llvm::Instruction::FPTrunc:
      return Opcode::FPTrunc;
    case llvm::Instruction::BitCast:
      return Opcode::BitCast;
    case llvm::Instruction::AddrSpaceCast:
      return Opcode::AddrSpaceCast;
    case llvm::Instruction::CastOpsEnd:
      llvm_unreachable("Bad CastOp!");
    }
    llvm_unreachable("Unhandled CastOp!");
  }
  /// Use Context::createCastInst(). Don't call the
  /// constructor directly.
  CastInst(llvm::CastInst *CI, Context &Ctx)
      : UnaryInstruction(ClassID::Cast, getCastOpcode(CI->getOpcode()), CI,
                         Ctx) {}
  friend Context; // for SBCastInstruction()

public:
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Type *getSrcTy() const;
  Type *getDestTy() const;
};

/// Instruction that can have a nneg flag (zext/uitofp).
class PossiblyNonNegInst : public CastInst {
public:
  bool hasNonNeg() const {
    return cast<llvm::PossiblyNonNegInst>(Val)->hasNonNeg();
  }
  void setNonNeg(bool B);
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From)) {
      switch (I->getOpcode()) {
      case Opcode::ZExt:
      case Opcode::UIToFP:
        return true;
      default:
        return false;
      }
    }
    return false;
  }
};

// Helper class to simplify stamping out CastInst subclasses.
template <Instruction::Opcode Op> class CastInstImpl : public CastInst {
public:
  static Value *create(Value *Src, Type *DestTy, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "") {
    return CastInst::create(DestTy, Op, Src, WhereIt, WhereBB, Ctx, Name);
  }
  static Value *create(Value *Src, Type *DestTy, Instruction *InsertBefore,
                       Context &Ctx, const Twine &Name = "") {
    return create(Src, DestTy, InsertBefore->getIterator(),
                  InsertBefore->getParent(), Ctx, Name);
  }
  static Value *create(Value *Src, Type *DestTy, BasicBlock *InsertAtEnd,
                       Context &Ctx, const Twine &Name = "") {
    return create(Src, DestTy, InsertAtEnd->end(), InsertAtEnd, Ctx, Name);
  }

  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From))
      return I->getOpcode() == Op;
    return false;
  }
};

class TruncInst final : public CastInstImpl<Instruction::Opcode::Trunc> {};
class ZExtInst final : public CastInstImpl<Instruction::Opcode::ZExt> {};
class SExtInst final : public CastInstImpl<Instruction::Opcode::SExt> {};
class FPTruncInst final : public CastInstImpl<Instruction::Opcode::FPTrunc> {};
class FPExtInst final : public CastInstImpl<Instruction::Opcode::FPExt> {};
class UIToFPInst final : public CastInstImpl<Instruction::Opcode::UIToFP> {};
class SIToFPInst final : public CastInstImpl<Instruction::Opcode::SIToFP> {};
class FPToUIInst final : public CastInstImpl<Instruction::Opcode::FPToUI> {};
class FPToSIInst final : public CastInstImpl<Instruction::Opcode::FPToSI> {};
class IntToPtrInst final : public CastInstImpl<Instruction::Opcode::IntToPtr> {
};
class PtrToIntInst final : public CastInstImpl<Instruction::Opcode::PtrToInt> {
};
class BitCastInst final : public CastInstImpl<Instruction::Opcode::BitCast> {};
class AddrSpaceCastInst final
    : public CastInstImpl<Instruction::Opcode::AddrSpaceCast> {
public:
  /// \Returns the pointer operand.
  Value *getPointerOperand() { return getOperand(0); }
  /// \Returns the pointer operand.
  const Value *getPointerOperand() const {
    return const_cast<AddrSpaceCastInst *>(this)->getPointerOperand();
  }
  /// \Returns the operand index of the pointer operand.
  static unsigned getPointerOperandIndex() { return 0u; }
  /// \Returns the address space of the pointer operand.
  unsigned getSrcAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }
  /// \Returns the address space of the result.
  unsigned getDestAddressSpace() const {
    return getType()->getPointerAddressSpace();
  }
};

class PHINode final : public SingleLLVMInstructionImpl<llvm::PHINode> {
  /// Use Context::createPHINode(). Don't call the constructor directly.
  PHINode(llvm::PHINode *PHI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::PHI, Opcode::PHI, PHI, Ctx) {}
  friend Context; // for PHINode()
  /// Helper for mapped_iterator.
  struct LLVMBBToBB {
    Context &Ctx;
    LLVMBBToBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock *operator()(llvm::BasicBlock *LLVMBB) const;
  };

public:
  static PHINode *create(Type *Ty, unsigned NumReservedValues,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);

  using const_block_iterator =
      mapped_iterator<llvm::PHINode::const_block_iterator, LLVMBBToBB>;

  const_block_iterator block_begin() const {
    LLVMBBToBB BBGetter(Ctx);
    return const_block_iterator(cast<llvm::PHINode>(Val)->block_begin(),
                                BBGetter);
  }
  const_block_iterator block_end() const {
    LLVMBBToBB BBGetter(Ctx);
    return const_block_iterator(cast<llvm::PHINode>(Val)->block_end(),
                                BBGetter);
  }
  iterator_range<const_block_iterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  op_range incoming_values() { return operands(); }

  const_op_range incoming_values() const { return operands(); }

  unsigned getNumIncomingValues() const {
    return cast<llvm::PHINode>(Val)->getNumIncomingValues();
  }
  Value *getIncomingValue(unsigned Idx) const;
  void setIncomingValue(unsigned Idx, Value *V);
  static unsigned getOperandNumForIncomingValue(unsigned Idx) {
    return llvm::PHINode::getOperandNumForIncomingValue(Idx);
  }
  static unsigned getIncomingValueNumForOperand(unsigned Idx) {
    return llvm::PHINode::getIncomingValueNumForOperand(Idx);
  }
  BasicBlock *getIncomingBlock(unsigned Idx) const;
  BasicBlock *getIncomingBlock(const Use &U) const;

  void setIncomingBlock(unsigned Idx, BasicBlock *BB);

  void addIncoming(Value *V, BasicBlock *BB);

  Value *removeIncomingValue(unsigned Idx);
  Value *removeIncomingValue(BasicBlock *BB);

  int getBasicBlockIndex(const BasicBlock *BB) const;
  Value *getIncomingValueForBlock(const BasicBlock *BB) const;

  Value *hasConstantValue() const;

  bool hasConstantOrUndefValue() const {
    return cast<llvm::PHINode>(Val)->hasConstantOrUndefValue();
  }
  bool isComplete() const { return cast<llvm::PHINode>(Val)->isComplete(); }
  void replaceIncomingBlockWith(const BasicBlock *Old, BasicBlock *New);
  void removeIncomingValueIf(function_ref<bool(unsigned)> Predicate);
  // TODO: Implement
  // void copyIncomingBlocks(iterator_range<const_block_iterator> BBRange,
  //                         uint32_t ToIdx = 0)
};

// Wraps a static function that takes a single Predicate parameter
// LLVMValType should be the type of the wrapped class
#define WRAP_STATIC_PREDICATE(FunctionName)                                    \
  static auto FunctionName(Predicate P) { return LLVMValType::FunctionName(P); }
// Wraps a member function that takes no parameters
// LLVMValType should be the type of the wrapped class
#define WRAP_MEMBER(FunctionName)                                              \
  auto FunctionName() const { return cast<LLVMValType>(Val)->FunctionName(); }
// Wraps both--a common idiom in the CmpInst classes
#define WRAP_BOTH(FunctionName)                                                \
  WRAP_STATIC_PREDICATE(FunctionName)                                          \
  WRAP_MEMBER(FunctionName)

class CmpInst : public SingleLLVMInstructionImpl<llvm::CmpInst> {
protected:
  using LLVMValType = llvm::CmpInst;
  /// Use Context::createCmpInst(). Don't call the constructor directly.
  CmpInst(llvm::CmpInst *CI, Context &Ctx, ClassID Id, Opcode Opc)
      : SingleLLVMInstructionImpl(Id, Opc, CI, Ctx) {}
  friend Context; // for CmpInst()
  static Value *createCommon(Value *Cond, Value *True, Value *False,
                             const Twine &Name, IRBuilder<> &Builder,
                             Context &Ctx);

public:
  using Predicate = llvm::CmpInst::Predicate;

  static CmpInst *create(Predicate Pred, Value *S1, Value *S2,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name = "");
  static CmpInst *createWithCopiedFlags(Predicate Pred, Value *S1, Value *S2,
                                        const Instruction *FlagsSource,
                                        Instruction *InsertBefore, Context &Ctx,
                                        const Twine &Name = "");
  void setPredicate(Predicate P);
  void swapOperands();

  WRAP_MEMBER(getPredicate);
  WRAP_BOTH(isFPPredicate);
  WRAP_BOTH(isIntPredicate);
  WRAP_STATIC_PREDICATE(getPredicateName);
  WRAP_BOTH(getInversePredicate);
  WRAP_BOTH(getOrderedPredicate);
  WRAP_BOTH(getUnorderedPredicate);
  WRAP_BOTH(getSwappedPredicate);
  WRAP_BOTH(isStrictPredicate);
  WRAP_BOTH(isNonStrictPredicate);
  WRAP_BOTH(getStrictPredicate);
  WRAP_BOTH(getNonStrictPredicate);
  WRAP_BOTH(getFlippedStrictnessPredicate);
  WRAP_MEMBER(isCommutative);
  WRAP_BOTH(isEquality);
  WRAP_BOTH(isRelational);
  WRAP_BOTH(isSigned);
  WRAP_BOTH(getSignedPredicate);
  WRAP_BOTH(getUnsignedPredicate);
  WRAP_BOTH(getFlippedSignednessPredicate);
  WRAP_BOTH(isTrueWhenEqual);
  WRAP_BOTH(isFalseWhenEqual);
  WRAP_BOTH(isUnsigned);
  WRAP_STATIC_PREDICATE(isOrdered);
  WRAP_STATIC_PREDICATE(isUnordered);

  static bool isImpliedTrueByMatchingCmp(Predicate Pred1, Predicate Pred2) {
    return llvm::CmpInst::isImpliedTrueByMatchingCmp(Pred1, Pred2);
  }
  static bool isImpliedFalseByMatchingCmp(Predicate Pred1, Predicate Pred2) {
    return llvm::CmpInst::isImpliedFalseByMatchingCmp(Pred1, Pred2);
  }

  /// Method for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ICmp ||
           From->getSubclassID() == ClassID::FCmp;
  }

  /// Create a result type for fcmp/icmp
  static Type *makeCmpResultType(Type *OpndType);

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class ICmpInst : public CmpInst {
  /// Use Context::createICmpInst(). Don't call the constructor directly.
  ICmpInst(llvm::ICmpInst *CI, Context &Ctx)
      : CmpInst(CI, Ctx, ClassID::ICmp, Opcode::ICmp) {}
  friend class Context; // For constructor.
  using LLVMValType = llvm::ICmpInst;

public:
  void swapOperands();

  WRAP_BOTH(getSignedPredicate);
  WRAP_BOTH(getUnsignedPredicate);
  WRAP_BOTH(isEquality);
  WRAP_MEMBER(isCommutative);
  WRAP_MEMBER(isRelational);
  WRAP_STATIC_PREDICATE(isGT);
  WRAP_STATIC_PREDICATE(isLT);
  WRAP_STATIC_PREDICATE(isGE);
  WRAP_STATIC_PREDICATE(isLE);

  static auto predicates() { return llvm::ICmpInst::predicates(); }
  static bool compare(const APInt &LHS, const APInt &RHS,
                      ICmpInst::Predicate Pred) {
    return llvm::ICmpInst::compare(LHS, RHS, Pred);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ICmp;
  }
};

class FCmpInst : public CmpInst {
  /// Use Context::createFCmpInst(). Don't call the constructor directly.
  FCmpInst(llvm::FCmpInst *CI, Context &Ctx)
      : CmpInst(CI, Ctx, ClassID::FCmp, Opcode::FCmp) {}
  friend class Context; // For constructor.
  using LLVMValType = llvm::FCmpInst;

public:
  void swapOperands();

  WRAP_BOTH(isEquality);
  WRAP_MEMBER(isCommutative);
  WRAP_MEMBER(isRelational);

  static auto predicates() { return llvm::FCmpInst::predicates(); }
  static bool compare(const APFloat &LHS, const APFloat &RHS,
                      FCmpInst::Predicate Pred) {
    return llvm::FCmpInst::compare(LHS, RHS, Pred);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::FCmp;
  }
};

#undef WRAP_STATIC_PREDICATE
#undef WRAP_MEMBER
#undef WRAP_BOTH

/// An LLLVM Instruction that has no SandboxIR equivalent class gets mapped to
/// an OpaqueInstr.
class OpaqueInst : public SingleLLVMInstructionImpl<llvm::Instruction> {
  OpaqueInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Opaque, Opcode::Opaque, I, Ctx) {}
  OpaqueInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::Opaque, I, Ctx) {}
  friend class Context; // For constructor.

public:
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
};

class Context {
protected:
  LLVMContext &LLVMCtx;
  friend class Type;        // For LLVMCtx.
  friend class PointerType; // For LLVMCtx.
  friend class CmpInst; // For LLVMCtx. TODO: cleanup when sandboxir::VectorType
                        // is complete
  friend class IntegerType;           // For LLVMCtx.
  friend class StructType;            // For LLVMCtx.
  friend class ::llvm::TargetExtType; // For LLVMCtx.
  friend class Region;                // For LLVMCtx.

  Tracker IRTracker;

  /// Maps LLVM Value to the corresponding sandboxir::Value. Owns all
  /// SandboxIR objects.
  DenseMap<llvm::Value *, std::unique_ptr<sandboxir::Value>>
      LLVMValueToValueMap;

  /// Maps an LLVM Module to the corresponding sandboxir::Module.
  DenseMap<llvm::Module *, std::unique_ptr<Module>> LLVMModuleToModuleMap;

  /// Type has a protected destructor to prohibit the user from managing the
  /// lifetime of the Type objects. Context is friend of Type, and this custom
  /// deleter can destroy Type.
  struct TypeDeleter {
    void operator()(Type *Ty) { delete Ty; }
  };
  /// Maps LLVM Type to the corresonding sandboxir::Type. Owns all Sandbox IR
  /// Type objects.
  DenseMap<llvm::Type *, std::unique_ptr<Type, TypeDeleter>> LLVMTypeToTypeMap;

  /// Remove \p V from the maps and returns the unique_ptr.
  std::unique_ptr<Value> detachLLVMValue(llvm::Value *V);
  /// Remove \p SBV from all SandboxIR maps and stop owning it. This effectively
  /// detaches \p V from the underlying IR.
  std::unique_ptr<Value> detach(Value *V);
  friend void Instruction::eraseFromParent(); // For detach().
  /// Take ownership of VPtr and store it in `LLVMValueToValueMap`.
  Value *registerValue(std::unique_ptr<Value> &&VPtr);
  friend class EraseFromParent; // For registerValue().
  /// This is the actual function that creates sandboxir values for \p V,
  /// and among others handles all instruction types.
  Value *getOrCreateValueInternal(llvm::Value *V, llvm::User *U = nullptr);
  /// Get or create a sandboxir::Argument for an existing LLVM IR \p LLVMArg.
  Argument *getOrCreateArgument(llvm::Argument *LLVMArg) {
    auto Pair = LLVMValueToValueMap.insert({LLVMArg, nullptr});
    auto It = Pair.first;
    if (Pair.second) {
      It->second = std::unique_ptr<Argument>(new Argument(LLVMArg, *this));
      return cast<Argument>(It->second.get());
    }
    return cast<Argument>(It->second.get());
  }
  /// Get or create a sandboxir::Value for an existing LLVM IR \p LLVMV.
  Value *getOrCreateValue(llvm::Value *LLVMV) {
    return getOrCreateValueInternal(LLVMV, 0);
  }
  /// Get or create a sandboxir::Constant from an existing LLVM IR \p LLVMC.
  Constant *getOrCreateConstant(llvm::Constant *LLVMC) {
    return cast<Constant>(getOrCreateValueInternal(LLVMC, 0));
  }
  // Friends for getOrCreateConstant().
#define DEF_CONST(ID, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

  /// Create a sandboxir::BasicBlock for an existing LLVM IR \p BB. This will
  /// also create all contents of the block.
  BasicBlock *createBasicBlock(llvm::BasicBlock *BB);
  friend class BasicBlock; // For getOrCreateValue().

  IRBuilder<ConstantFolder> LLVMIRBuilder;
  auto &getLLVMIRBuilder() { return LLVMIRBuilder; }

  VAArgInst *createVAArgInst(llvm::VAArgInst *SI);
  friend VAArgInst; // For createVAArgInst()
  FreezeInst *createFreezeInst(llvm::FreezeInst *SI);
  friend FreezeInst; // For createFreezeInst()
  FenceInst *createFenceInst(llvm::FenceInst *SI);
  friend FenceInst; // For createFenceInst()
  SelectInst *createSelectInst(llvm::SelectInst *SI);
  friend SelectInst; // For createSelectInst()
  InsertElementInst *createInsertElementInst(llvm::InsertElementInst *IEI);
  friend InsertElementInst; // For createInsertElementInst()
  ExtractElementInst *createExtractElementInst(llvm::ExtractElementInst *EEI);
  friend ExtractElementInst; // For createExtractElementInst()
  ShuffleVectorInst *createShuffleVectorInst(llvm::ShuffleVectorInst *SVI);
  friend ShuffleVectorInst; // For createShuffleVectorInst()
  ExtractValueInst *createExtractValueInst(llvm::ExtractValueInst *IVI);
  friend ExtractValueInst; // For createExtractValueInst()
  InsertValueInst *createInsertValueInst(llvm::InsertValueInst *IVI);
  friend InsertValueInst; // For createInsertValueInst()
  BranchInst *createBranchInst(llvm::BranchInst *I);
  friend BranchInst; // For createBranchInst()
  LoadInst *createLoadInst(llvm::LoadInst *LI);
  friend LoadInst; // For createLoadInst()
  StoreInst *createStoreInst(llvm::StoreInst *SI);
  friend StoreInst; // For createStoreInst()
  ReturnInst *createReturnInst(llvm::ReturnInst *I);
  friend ReturnInst; // For createReturnInst()
  CallInst *createCallInst(llvm::CallInst *I);
  friend CallInst; // For createCallInst()
  InvokeInst *createInvokeInst(llvm::InvokeInst *I);
  friend InvokeInst; // For createInvokeInst()
  CallBrInst *createCallBrInst(llvm::CallBrInst *I);
  friend CallBrInst; // For createCallBrInst()
  LandingPadInst *createLandingPadInst(llvm::LandingPadInst *I);
  friend LandingPadInst; // For createLandingPadInst()
  CatchPadInst *createCatchPadInst(llvm::CatchPadInst *I);
  friend CatchPadInst; // For createCatchPadInst()
  CleanupPadInst *createCleanupPadInst(llvm::CleanupPadInst *I);
  friend CleanupPadInst; // For createCleanupPadInst()
  CatchReturnInst *createCatchReturnInst(llvm::CatchReturnInst *I);
  friend CatchReturnInst; // For createCatchReturnInst()
  CleanupReturnInst *createCleanupReturnInst(llvm::CleanupReturnInst *I);
  friend CleanupReturnInst; // For createCleanupReturnInst()
  GetElementPtrInst *createGetElementPtrInst(llvm::GetElementPtrInst *I);
  friend GetElementPtrInst; // For createGetElementPtrInst()
  CatchSwitchInst *createCatchSwitchInst(llvm::CatchSwitchInst *I);
  friend CatchSwitchInst; // For createCatchSwitchInst()
  ResumeInst *createResumeInst(llvm::ResumeInst *I);
  friend ResumeInst; // For createResumeInst()
  SwitchInst *createSwitchInst(llvm::SwitchInst *I);
  friend SwitchInst; // For createSwitchInst()
  UnaryOperator *createUnaryOperator(llvm::UnaryOperator *I);
  friend UnaryOperator; // For createUnaryOperator()
  BinaryOperator *createBinaryOperator(llvm::BinaryOperator *I);
  friend BinaryOperator; // For createBinaryOperator()
  AtomicRMWInst *createAtomicRMWInst(llvm::AtomicRMWInst *I);
  friend AtomicRMWInst; // For createAtomicRMWInst()
  AtomicCmpXchgInst *createAtomicCmpXchgInst(llvm::AtomicCmpXchgInst *I);
  friend AtomicCmpXchgInst; // For createAtomicCmpXchgInst()
  AllocaInst *createAllocaInst(llvm::AllocaInst *I);
  friend AllocaInst; // For createAllocaInst()
  CastInst *createCastInst(llvm::CastInst *I);
  friend CastInst; // For createCastInst()
  PHINode *createPHINode(llvm::PHINode *I);
  friend PHINode; // For createPHINode()
  UnreachableInst *createUnreachableInst(llvm::UnreachableInst *UI);
  friend UnreachableInst; // For createUnreachableInst()
  CmpInst *createCmpInst(llvm::CmpInst *I);
  friend CmpInst; // For createCmpInst()
  ICmpInst *createICmpInst(llvm::ICmpInst *I);
  friend ICmpInst; // For createICmpInst()
  FCmpInst *createFCmpInst(llvm::FCmpInst *I);
  friend FCmpInst; // For createFCmpInst()

public:
  Context(LLVMContext &LLVMCtx)
      : LLVMCtx(LLVMCtx), IRTracker(*this),
        LLVMIRBuilder(LLVMCtx, ConstantFolder()) {}

  Tracker &getTracker() { return IRTracker; }
  /// Convenience function for `getTracker().save()`
  void save() { IRTracker.save(); }
  /// Convenience function for `getTracker().revert()`
  void revert() { IRTracker.revert(); }
  /// Convenience function for `getTracker().accept()`
  void accept() { IRTracker.accept(); }

  sandboxir::Value *getValue(llvm::Value *V) const;
  const sandboxir::Value *getValue(const llvm::Value *V) const {
    return getValue(const_cast<llvm::Value *>(V));
  }

  Module *getModule(llvm::Module *LLVMM) const;

  Module *getOrCreateModule(llvm::Module *LLVMM);

  Type *getType(llvm::Type *LLVMTy) {
    if (LLVMTy == nullptr)
      return nullptr;
    auto Pair = LLVMTypeToTypeMap.insert({LLVMTy, nullptr});
    auto It = Pair.first;
    if (Pair.second)
      It->second = std::unique_ptr<Type, TypeDeleter>(new Type(LLVMTy, *this));
    return It->second.get();
  }

  /// Create a sandboxir::Function for an existing LLVM IR \p F, including all
  /// blocks and instructions.
  /// This is the main API function for creating Sandbox IR.
  /// Note: this will not fully populate its parent module. The only globals
  /// that will be available are those used within the function.
  Function *createFunction(llvm::Function *F);

  /// Create a sandboxir::Module corresponding to \p LLVMM.
  Module *createModule(llvm::Module *LLVMM);

  /// \Returns the number of values registered with Context.
  size_t getNumValues() const { return LLVMValueToValueMap.size(); }
};

class Function : public GlobalWithNodeAPI<Function, llvm::Function,
                                          GlobalObject, llvm::GlobalObject> {
  /// Helper for mapped_iterator.
  struct LLVMBBToBB {
    Context &Ctx;
    LLVMBBToBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock &operator()(llvm::BasicBlock &LLVMBB) const {
      return *cast<BasicBlock>(Ctx.getValue(&LLVMBB));
    }
  };
  /// Use Context::createFunction() instead.
  Function(llvm::Function *F, sandboxir::Context &Ctx)
      : GlobalWithNodeAPI(ClassID::Function, F, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  Module *getParent() {
    return Ctx.getModule(cast<llvm::Function>(Val)->getParent());
  }

  Argument *getArg(unsigned Idx) const {
    llvm::Argument *Arg = cast<llvm::Function>(Val)->getArg(Idx);
    return cast<Argument>(Ctx.getValue(Arg));
  }

  size_t arg_size() const { return cast<llvm::Function>(Val)->arg_size(); }
  bool arg_empty() const { return cast<llvm::Function>(Val)->arg_empty(); }

  using iterator = mapped_iterator<llvm::Function::iterator, LLVMBBToBB>;
  iterator begin() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->begin(), BBGetter);
  }
  iterator end() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->end(), BBGetter);
  }
  FunctionType *getFunctionType() const;

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_SANDBOXIR_SANDBOXIR_H
