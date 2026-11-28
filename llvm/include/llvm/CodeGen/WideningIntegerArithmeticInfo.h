//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WIDENINGINTEGERARITHMETICINFO_H
#define LLVM_CODEGEN_WIDENINGINTEGERARITHMETICINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

inline constexpr unsigned FILL_INST_OPC = Instruction::OtherOpsEnd;
inline constexpr unsigned CONSTANT_INT_OPC = FILL_INST_OPC + 1;

class Value;

enum IntegerFillType { SIGN = 0, ZEROS, ANYTHING, UNDEFINED, PENDING };

enum WIAKind {
  WIAK_BINOP = 0,
  WIAK_PHI,
  WIAK_UNOP,
  WIAK_FILL,
  WIAK_WIDEN,
  WIAK_WIDEN_GARBAGE,
  WIAK_NARROW,
  WIAK_DROP_EXT,
  WIAK_DROP_LOCOPY,
  WIAK_DROP_LOIGNORE,
  WIAK_EXTLO,
  WIAK_SUBSUME_FILL,
  WIAK_SUBSUME_INDEX,
  WIAK_NATURAL,
  WIAK_STORE,
  WIAK_CONSTANT,
  WIAK_LIT,
  WIAK_LOAD,
  WIAK_UNKNOWN,
  WIAK_ICMP,
  WIAK_NOOP,
  WIAK_SWITCH,
  WIAK_RET,
  WIAK_SELECT,
};

inline constexpr const char *WIAK_NAMES_VEC[] = {
    "WIAK_BINOP",         "WIAK_PHI",      "WIAK_UNOP",
    "WIAK_FILL",          "WIAK_WIDEN",    "WIAK_WIDEN_GARBAGE",
    "WIAK_NARROW",        "WIAK_DROP_EXT", "WIAK_DROP_LOCOPY",
    "WIAK_DROP_LOIGNORE", "WIAK_EXTLO",    "WIAK_SUBSUME_FILL",
    "WIAK_SUBSUME_INDEX", "WIAK_NATURAL",  "WIAK_STORE",
    "WIAK_CONSTANT",      "WIAK_LIT",      "WIAK_LOAD",
    "WIAK_UNKNOWN",       "WIAK_ICMP",     "WIAK_NOOP",
    "WIAK_SWITCH",        "WIAK_RET",      "WIAK_SELECT"};

// This class is used to specify all the arguments that a solution
// of WideningIntegerArithmetic class requires.
class WideningIntegerSolutionInfo {
private:
  // We care only about integer arithmetic instructions
  unsigned Opcode;
  // The new Opcode that has *this* as an operand
  unsigned NewOpcode;
  // Fill type for the upper bits of the expression
  // can be zero, one or garbage
  IntegerFillType FillType;     // τ
  unsigned short FillTypeWidth; // n, where τ[n] indicates
                                // the lower bits of the instruction
                                // with fillType τ.
  // The current width of the instruction's destination operand
  unsigned short Width; // w
  // The updated width of the instruction's destination operand
  unsigned short UpdatedWidth;
  // The cost of the solution
  InstructionCost Cost;
  // Pointer to the Value/Value that mapped to this solutionInfo
  Value *V;
  // What Kind is this Sol is it a Binary Operator? a ICmp instruction?
  WIAKind Kind;
  SmallVector<WideningIntegerSolutionInfo *, 4> Operands;
  unsigned short UpdatedICmpWidth = 0;

public:
  WideningIntegerSolutionInfo() = delete;
  ~WideningIntegerSolutionInfo() = default;
  WideningIntegerSolutionInfo(unsigned Opcode_, unsigned NewOpcode_,
                              IntegerFillType FillType_,
                              unsigned short FillTypeWidth_,
                              unsigned short Width_,
                              unsigned short UpdatedWidth_,
                              InstructionCost Cost_, WIAKind Kind_, Value *V_)
      : Opcode(Opcode_), NewOpcode(NewOpcode_), FillType(FillType_),
        FillTypeWidth(FillTypeWidth_), Width(Width_),
        UpdatedWidth(UpdatedWidth_), Cost(Cost_), V(V_), Kind(Kind_) {}

  // We don't need copy or move constructors we use BumpPtrAllocator
  // and we copy pointers into the collections, that's cheap.
  WideningIntegerSolutionInfo(WideningIntegerSolutionInfo const &) = delete;
  WideningIntegerSolutionInfo &
  operator=(WideningIntegerSolutionInfo const &) = delete;

  WideningIntegerSolutionInfo(WideningIntegerSolutionInfo &&) = delete;
  WideningIntegerSolutionInfo &
  operator=(WideningIntegerSolutionInfo &&) = delete;

  unsigned getOpcode(void) const { return Opcode; }
  void setOpcode(unsigned Opcode_) { Opcode = Opcode_; }

  unsigned getNewOpcode(void) const { return NewOpcode; }
  void setNewOpcode(unsigned NewOpcode_) { NewOpcode = NewOpcode_; }

  InstructionCost getCost(void) const { return Cost; }
  void setCost(InstructionCost Cost_) { Cost = Cost_; }
  void incrementCost() { ++Cost; }

  IntegerFillType getFillType(void) const { return FillType; }
  void setFillType(IntegerFillType FillType_) { FillType = FillType_; }

  unsigned short getWidth(void) const { return Width; }
  void setWidth(unsigned short Width_) { Width = Width_; }

  unsigned short getUpdatedWidth(void) const { return UpdatedWidth; }

  void setUpdatedWidth(unsigned short UpdatedWidth_) {
    UpdatedWidth = UpdatedWidth_;
  }

  unsigned short getUpdatedICmpWidth() const { return UpdatedICmpWidth; }

  void setUpdatedICmpWidth(unsigned short ICmpW_) { UpdatedICmpWidth = ICmpW_; }

  // For the ICmp instruction we only optimize for the compare width,
  // not the return width.
  unsigned short getEffectiveWidth() const {
    return Kind == WIAK_ICMP ? UpdatedICmpWidth : UpdatedWidth;
  }

  void setKind(WIAKind Kind_) { Kind = Kind_; }
  WIAKind getKind() const { return Kind; }

  void setOperands(ArrayRef<WideningIntegerSolutionInfo *> Ops) {
    Operands.assign(Ops.begin(), Ops.end());
  }

  ArrayRef<WideningIntegerSolutionInfo *> getOperands(void) const {
    return Operands;
  }
  void addOperand(WideningIntegerSolutionInfo *Sol) { Operands.push_back(Sol); }

  const WideningIntegerSolutionInfo *getOperand(short int i) const {
    return Operands[i];
  }

  unsigned short getFillTypeWidth(void) const { return FillTypeWidth; }

  void setFillTypeWidth(unsigned short FillType_) { FillTypeWidth = FillType_; }

  void setValue(Value *V_) { V = V_; }

  Value *getValue(void) const { return V; }

  // Returns 0 if no one is redudant
  // Returns -1 if b is redundant given solution this
  // Returns 1 if "this" is redudant given solution b
  inline int IsRedudant(const WideningIntegerSolutionInfo &b) const noexcept;

}; // WideningIntegerSolutionInfo

inline raw_ostream &operator<<(raw_ostream &out,
                               WideningIntegerSolutionInfo const &Sol) {
  auto GetOpcodeName = [](unsigned Opcode) -> const char * {
    if (Opcode == FILL_INST_OPC)
      return "FillInst";
    if (Opcode == CONSTANT_INT_OPC)
      return "ConstantInt";
    if (Opcode == 0)
      return "Empty Instr";
    if (Opcode >= Instruction::TermOpsBegin &&
        Opcode < Instruction::OtherOpsEnd)
      return Instruction::getOpcodeName(Opcode);
    return "UnknownOpc";
  };

  out << "\tOldOpcode: " << GetOpcodeName(Sol.getOpcode()) << '\n';
  out << "\tOldOpcode in number: " << Sol.getOpcode() << '\n';
  out << "\tNewOpcode: " << GetOpcodeName(Sol.getNewOpcode()) << '\n';
  out << "\tWIA Kind : " << WIAK_NAMES_VEC[Sol.getKind()] << '\n';
  out << "\tFillType: " << Sol.getFillType() << '\n';
  out << "\tFillTypeWidth: " << Sol.getFillTypeWidth() << '\n';
  out << "\tOldWidth: " << Sol.getWidth() << '\n';
  out << "\tUpdatedWidth: " << Sol.getUpdatedWidth() << '\n';
  out << "\tCost : " << Sol.getCost() << '\n';
  if (Sol.getKind() == WIAK_ICMP) {
    out << "\tUpdatedIcmpWidth : " << Sol.getUpdatedICmpWidth() << '\n';
  }

  return out;
}

// Returns 0 if no one is redudant
// Returns -1 if b is redundant given solution this
// Returns 1 if "this" is redudant given solution b
inline int WideningIntegerSolutionInfo::IsRedudant(
    const WideningIntegerSolutionInfo &b) const noexcept {

  int n1 = FillTypeWidth;
  auto c1 = Cost;
  int n2 = b.getFillTypeWidth();
  auto c2 = b.getCost();
  int FillType1 = FillType;
  int FillType2 = b.getFillType();

  int w1 = getEffectiveWidth();
  int w2 = b.getEffectiveWidth();
  // different updatedWidth or different FillType or different Inst no
  // redundant.
  if (w1 != w2 || FillType1 != FillType2 || b.getValue() != this->getValue()) {
    return 0;
  }

  // Derived from the cc paper
  // We keep this if it has less or equal FillTypeWidth
  // with less or equal cost.
  // Otherwise we try to keep the b solution.
  const bool this_leq_b = (n1 <= n2) && (c1 <= c2);
  const bool b_leq_this = (n2 <= n1) && (c2 <= c1);

  // keep this, drop b.
  if (this_leq_b && b_leq_this)
    return -1;

  // b dominates this.
  if (b_leq_this)
    return 1;
  // this dominates b.
  if (this_leq_b)
    return -1;

  return 0;
}

} // namespace llvm

#endif // LLVM_CODEGEN_WIDENINGINTEGERARITHMETICINFO_H
