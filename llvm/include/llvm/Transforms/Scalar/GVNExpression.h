//======- GVNExpression.h - GVN Expression classes --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// The header file for the GVN pass that contains expression handling
/// classes
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVNEXPRESSION_H
#define LLVM_TRANSFORMS_SCALAR_GVNEXPRESSION_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

namespace llvm {

namespace GVNExpression {

enum ExpressionType {
  ET_Base,
  ET_Constant,
  ET_Variable,
  ET_Unknown,
  ET_BasicStart,
  ET_Basic,
  ET_Call,
  ET_AggregateValue,
  ET_Phi,
  ET_Load,
  ET_Store,
  ET_BasicEnd
};

class Expression {
private:
  ExpressionType EType;
  unsigned Opcode;

public:
  Expression(ExpressionType ET = ET_Base, unsigned O = ~2U)
      : EType(ET), Opcode(O) {}
  Expression(const Expression &) = delete;
  Expression &operator=(const Expression &) = delete;
  virtual ~Expression();

  static unsigned getEmptyKey() { return ~0U; }
  static unsigned getTombstoneKey() { return ~1U; }
  bool operator!=(const Expression &Other) const { return !(*this == Other); }
  bool operator==(const Expression &Other) const {
    if (getOpcode() != Other.getOpcode())
      return false;
    if (getOpcode() == getEmptyKey() || getOpcode() == getTombstoneKey())
      return true;
    // Compare the expression type for anything but load and store.
    // For load and store we set the opcode to zero.
    // This is needed for load coercion.
    if (getExpressionType() != ET_Load && getExpressionType() != ET_Store &&
        getExpressionType() != Other.getExpressionType())
      return false;

    return equals(Other);
  }

  virtual bool equals(const Expression &Other) const { return true; }

  unsigned getOpcode() const { return Opcode; }
  void setOpcode(unsigned opcode) { Opcode = opcode; }
  ExpressionType getExpressionType() const { return EType; }

  virtual hash_code getHashValue() const {
    return hash_combine(getExpressionType(), getOpcode());
  }

  //
  // Debugging support
  //
  virtual void printInternal(raw_ostream &OS, bool PrintEType) const {
    if (PrintEType)
      OS << "etype = " << getExpressionType() << ",";
    OS << "opcode = " << getOpcode() << ", ";
  }

  void print(raw_ostream &OS) const {
    OS << "{ ";
    printInternal(OS, true);
    OS << "}";
  }

  void dump() const { print(dbgs()); }
};

inline raw_ostream &operator<<(raw_ostream &OS, const Expression &E) {
  E.print(OS);
  return OS;
}

class BasicExpression : public Expression {
private:
  typedef ArrayRecycler<Value *> RecyclerType;
  typedef RecyclerType::Capacity RecyclerCapacity;
  Value **Operands;
  unsigned MaxOperands;
  unsigned NumOperands;
  Type *ValueType;

public:
  BasicExpression(unsigned NumOperands)
      : BasicExpression(NumOperands, ET_Basic) {}
  BasicExpression(unsigned NumOperands, ExpressionType ET)
      : Expression(ET), Operands(nullptr), MaxOperands(NumOperands),
        NumOperands(0), ValueType(nullptr) {}
  BasicExpression() = delete;
  BasicExpression(const BasicExpression &) = delete;
  BasicExpression &operator=(const BasicExpression &) = delete;
  ~BasicExpression() override;

  static bool classof(const Expression *EB) {
    ExpressionType ET = EB->getExpressionType();
    return ET > ET_BasicStart && ET < ET_BasicEnd;
  }

  /// \brief Swap two operands. Used during GVN to put commutative operands in
  /// order.
  void swapOperands(unsigned First, unsigned Second) {
    std::swap(Operands[First], Operands[Second]);
  }

  Value *getOperand(unsigned N) const {
    assert(Operands && "Operands not allocated");
    assert(N < NumOperands && "Operand out of range");
    return Operands[N];
  }

  void setOperand(unsigned N, Value *V) {
    assert(Operands && "Operands not allocated before setting");
    assert(N < NumOperands && "Operand out of range");
    Operands[N] = V;
  }

  unsigned getNumOperands() const { return NumOperands; }

  typedef Value **op_iterator;
  typedef Value *const *const_op_iterator;
  op_iterator op_begin() { return Operands; }
  op_iterator op_end() { return Operands + NumOperands; }
  const_op_iterator op_begin() const { return Operands; }
  const_op_iterator op_end() const { return Operands + NumOperands; }
  iterator_range<op_iterator> operands() {
    return iterator_range<op_iterator>(op_begin(), op_end());
  }
  iterator_range<const_op_iterator> operands() const {
    return iterator_range<const_op_iterator>(op_begin(), op_end());
  }

  void op_push_back(Value *Arg) {
    assert(NumOperands < MaxOperands && "Tried to add too many operands");
    assert(Operands && "Operandss not allocated before pushing");
    Operands[NumOperands++] = Arg;
  }
  bool op_empty() const { return getNumOperands() == 0; }

  void allocateOperands(RecyclerType &Recycler, BumpPtrAllocator &Allocator) {
    assert(!Operands && "Operands already allocated");
    Operands = Recycler.allocate(RecyclerCapacity::get(MaxOperands), Allocator);
  }
  void deallocateOperands(RecyclerType &Recycler) {
    Recycler.deallocate(RecyclerCapacity::get(MaxOperands), Operands);
  }

  void setType(Type *T) { ValueType = T; }
  Type *getType() const { return ValueType; }

  bool equals(const Expression &Other) const override {
    if (getOpcode() != Other.getOpcode())
      return false;

    const auto &OE = cast<BasicExpression>(Other);
    return getType() == OE.getType() && NumOperands == OE.NumOperands &&
           std::equal(op_begin(), op_end(), OE.op_begin());
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), getOpcode(), ValueType,
                        hash_combine_range(op_begin(), op_end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeBasic, ";

    this->Expression::printInternal(OS, false);
    OS << "operands = {";
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      OS << "[" << i << "] = ";
      Operands[i]->printAsOperand(OS);
      OS << "  ";
    }
    OS << "} ";
  }
};

class op_inserter
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
private:
  typedef BasicExpression Container;
  Container *BE;

public:
  explicit op_inserter(BasicExpression &E) : BE(&E) {}
  explicit op_inserter(BasicExpression *E) : BE(E) {}

  op_inserter &operator=(Value *val) {
    BE->op_push_back(val);
    return *this;
  }
  op_inserter &operator*() { return *this; }
  op_inserter &operator++() { return *this; }
  op_inserter &operator++(int) { return *this; }
};

class CallExpression final : public BasicExpression {
private:
  CallInst *Call;
  MemoryAccess *DefiningAccess;

public:
  CallExpression(unsigned NumOperands, CallInst *C, MemoryAccess *DA)
      : BasicExpression(NumOperands, ET_Call), Call(C), DefiningAccess(DA) {}
  CallExpression() = delete;
  CallExpression(const CallExpression &) = delete;
  CallExpression &operator=(const CallExpression &) = delete;
  ~CallExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Call;
  }

  bool equals(const Expression &Other) const override {
    if (!this->BasicExpression::equals(Other))
      return false;
    const auto &OE = cast<CallExpression>(Other);
    return DefiningAccess == OE.DefiningAccess;
  }

  hash_code getHashValue() const override {
    return hash_combine(this->BasicExpression::getHashValue(), DefiningAccess);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeCall, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents call at " << Call;
  }
};

class LoadExpression final : public BasicExpression {
private:
  LoadInst *Load;
  MemoryAccess *DefiningAccess;
  unsigned Alignment;

public:
  LoadExpression(unsigned NumOperands, LoadInst *L, MemoryAccess *DA)
      : LoadExpression(ET_Load, NumOperands, L, DA) {}
  LoadExpression(enum ExpressionType EType, unsigned NumOperands, LoadInst *L,
                 MemoryAccess *DA)
      : BasicExpression(NumOperands, EType), Load(L), DefiningAccess(DA) {
    Alignment = L ? L->getAlignment() : 0;
  }
  LoadExpression() = delete;
  LoadExpression(const LoadExpression &) = delete;
  LoadExpression &operator=(const LoadExpression &) = delete;
  ~LoadExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Load;
  }

  LoadInst *getLoadInst() const { return Load; }
  void setLoadInst(LoadInst *L) { Load = L; }

  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }
  void setDefiningAccess(MemoryAccess *MA) { DefiningAccess = MA; }
  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Align) { Alignment = Align; }

  bool equals(const Expression &Other) const override;

  hash_code getHashValue() const override {
    return hash_combine(getOpcode(), getType(), DefiningAccess,
                        hash_combine_range(op_begin(), op_end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeLoad, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Load at " << Load;
    OS << " with DefiningAccess " << *DefiningAccess;
  }
};

class StoreExpression final : public BasicExpression {
private:
  StoreInst *Store;
  Value *StoredValue;
  MemoryAccess *DefiningAccess;

public:
  StoreExpression(unsigned NumOperands, StoreInst *S, Value *StoredValue,
                  MemoryAccess *DA)
      : BasicExpression(NumOperands, ET_Store), Store(S),
        StoredValue(StoredValue), DefiningAccess(DA) {}
  StoreExpression() = delete;
  StoreExpression(const StoreExpression &) = delete;
  StoreExpression &operator=(const StoreExpression &) = delete;
  ~StoreExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Store;
  }

  StoreInst *getStoreInst() const { return Store; }
  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }
  Value *getStoredValue() const { return StoredValue; }

  bool equals(const Expression &Other) const override;

  hash_code getHashValue() const override {
    // This deliberately does not include the stored value we compare it as part
    // of equals, and only against other stores.
    return hash_combine(getOpcode(), getType(), DefiningAccess,
                        hash_combine_range(op_begin(), op_end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeStore, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Store at " << Store;
    OS << " with DefiningAccess " << *DefiningAccess;
  }
};

class AggregateValueExpression final : public BasicExpression {
private:
  unsigned MaxIntOperands;
  unsigned NumIntOperands;
  unsigned *IntOperands;

public:
  AggregateValueExpression(unsigned NumOperands, unsigned NumIntOperands)
      : BasicExpression(NumOperands, ET_AggregateValue),
        MaxIntOperands(NumIntOperands), NumIntOperands(0),
        IntOperands(nullptr) {}
  AggregateValueExpression() = delete;
  AggregateValueExpression(const AggregateValueExpression &) = delete;
  AggregateValueExpression &
  operator=(const AggregateValueExpression &) = delete;
  ~AggregateValueExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_AggregateValue;
  }

  typedef unsigned *int_arg_iterator;
  typedef const unsigned *const_int_arg_iterator;

  int_arg_iterator int_op_begin() { return IntOperands; }
  int_arg_iterator int_op_end() { return IntOperands + NumIntOperands; }
  const_int_arg_iterator int_op_begin() const { return IntOperands; }
  const_int_arg_iterator int_op_end() const {
    return IntOperands + NumIntOperands;
  }
  unsigned int_op_size() const { return NumIntOperands; }
  bool int_op_empty() const { return NumIntOperands == 0; }
  void int_op_push_back(unsigned IntOperand) {
    assert(NumIntOperands < MaxIntOperands &&
           "Tried to add too many int operands");
    assert(IntOperands && "Operands not allocated before pushing");
    IntOperands[NumIntOperands++] = IntOperand;
  }

  virtual void allocateIntOperands(BumpPtrAllocator &Allocator) {
    assert(!IntOperands && "Operands already allocated");
    IntOperands = Allocator.Allocate<unsigned>(MaxIntOperands);
  }

  bool equals(const Expression &Other) const override {
    if (!this->BasicExpression::equals(Other))
      return false;
    const AggregateValueExpression &OE = cast<AggregateValueExpression>(Other);
    return NumIntOperands == OE.NumIntOperands &&
           std::equal(int_op_begin(), int_op_end(), OE.int_op_begin());
  }

  hash_code getHashValue() const override {
    return hash_combine(this->BasicExpression::getHashValue(),
                        hash_combine_range(int_op_begin(), int_op_end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeAggregateValue, ";
    this->BasicExpression::printInternal(OS, false);
    OS << ", intoperands = {";
    for (unsigned i = 0, e = int_op_size(); i != e; ++i) {
      OS << "[" << i << "] = " << IntOperands[i] << "  ";
    }
    OS << "}";
  }
};

class int_op_inserter
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
private:
  typedef AggregateValueExpression Container;
  Container *AVE;

public:
  explicit int_op_inserter(AggregateValueExpression &E) : AVE(&E) {}
  explicit int_op_inserter(AggregateValueExpression *E) : AVE(E) {}

  int_op_inserter &operator=(unsigned int val) {
    AVE->int_op_push_back(val);
    return *this;
  }
  int_op_inserter &operator*() { return *this; }
  int_op_inserter &operator++() { return *this; }
  int_op_inserter &operator++(int) { return *this; }
};

class PHIExpression final : public BasicExpression {
private:
  BasicBlock *BB;

public:
  PHIExpression(unsigned NumOperands, BasicBlock *B)
      : BasicExpression(NumOperands, ET_Phi), BB(B) {}
  PHIExpression() = delete;
  PHIExpression(const PHIExpression &) = delete;
  PHIExpression &operator=(const PHIExpression &) = delete;
  ~PHIExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Phi;
  }

  bool equals(const Expression &Other) const override {
    if (!this->BasicExpression::equals(Other))
      return false;
    const PHIExpression &OE = cast<PHIExpression>(Other);
    return BB == OE.BB;
  }

  hash_code getHashValue() const override {
    return hash_combine(this->BasicExpression::getHashValue(), BB);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypePhi, ";
    this->BasicExpression::printInternal(OS, false);
    OS << "bb = " << BB;
  }
};

class VariableExpression final : public Expression {
private:
  Value *VariableValue;

public:
  VariableExpression(Value *V) : Expression(ET_Variable), VariableValue(V) {}
  VariableExpression() = delete;
  VariableExpression(const VariableExpression &) = delete;
  VariableExpression &operator=(const VariableExpression &) = delete;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Variable;
  }

  Value *getVariableValue() const { return VariableValue; }
  void setVariableValue(Value *V) { VariableValue = V; }

  bool equals(const Expression &Other) const override {
    const VariableExpression &OC = cast<VariableExpression>(Other);
    return VariableValue == OC.VariableValue;
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), VariableValue->getType(),
                        VariableValue);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeVariable, ";
    this->Expression::printInternal(OS, false);
    OS << " variable = " << *VariableValue;
  }
};

class ConstantExpression final : public Expression {
private:
  Constant *ConstantValue = nullptr;

public:
  ConstantExpression() : Expression(ET_Constant) {}
  ConstantExpression(Constant *constantValue)
      : Expression(ET_Constant), ConstantValue(constantValue) {}
  ConstantExpression(const ConstantExpression &) = delete;
  ConstantExpression &operator=(const ConstantExpression &) = delete;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Constant;
  }

  Constant *getConstantValue() const { return ConstantValue; }
  void setConstantValue(Constant *V) { ConstantValue = V; }

  bool equals(const Expression &Other) const override {
    const ConstantExpression &OC = cast<ConstantExpression>(Other);
    return ConstantValue == OC.ConstantValue;
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), ConstantValue->getType(),
                        ConstantValue);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeConstant, ";
    this->Expression::printInternal(OS, false);
    OS << " constant = " << *ConstantValue;
  }
};

class UnknownExpression final : public Expression {
private:
  Instruction *Inst;

public:
  UnknownExpression(Instruction *I) : Expression(ET_Unknown), Inst(I) {}
  UnknownExpression() = delete;
  UnknownExpression(const UnknownExpression &) = delete;
  UnknownExpression &operator=(const UnknownExpression &) = delete;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Unknown;
  }

  Instruction *getInstruction() const { return Inst; }
  void setInstruction(Instruction *I) { Inst = I; }

  bool equals(const Expression &Other) const override {
    const auto &OU = cast<UnknownExpression>(Other);
    return Inst == OU.Inst;
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), Inst);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS, bool PrintEType) const override {
    if (PrintEType)
      OS << "ExpressionTypeUnknown, ";
    this->Expression::printInternal(OS, false);
    OS << " inst = " << *Inst;
  }
};

} // end namespace GVNExpression

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVNEXPRESSION_H
