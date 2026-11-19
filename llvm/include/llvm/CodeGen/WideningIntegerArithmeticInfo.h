#ifndef LLVM_CODEGEN_WIDENINGINTEGERARITHMETICINFO_H
#define LLVM_CODEGEN_WIDENINGINTEGERARITHMETICINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
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

public:
protected:
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

private:
  WIAKind Kind;
  SmallVector<WideningIntegerSolutionInfo *, 4> Operands;

public:
  WideningIntegerSolutionInfo() = delete;
  ~WideningIntegerSolutionInfo() = default;
  WideningIntegerSolutionInfo(WIAKind Kind_) : Kind(Kind_) {}
  WideningIntegerSolutionInfo(unsigned Opcode_, unsigned NewOpcode_,
                              IntegerFillType FillType_,
                              unsigned short FillTypeWidth_,
                              unsigned short Width_,
                              unsigned short UpdatedWidth_,
                              InstructionCost Cost_, WIAKind Kind_, Value *V_)
      : Opcode(Opcode_), NewOpcode(NewOpcode_), FillType(FillType_),
        FillTypeWidth(FillTypeWidth_), Width(Width_),
        UpdatedWidth(UpdatedWidth_), Cost(Cost_), V(V_), Kind(Kind_) {}

  WideningIntegerSolutionInfo(WideningIntegerSolutionInfo const &) = delete;
  WideningIntegerSolutionInfo &
  operator=(WideningIntegerSolutionInfo const &) = delete;

  WideningIntegerSolutionInfo(WideningIntegerSolutionInfo &&) = delete;
  WideningIntegerSolutionInfo
  operator=(WideningIntegerSolutionInfo &&) = delete;

  unsigned getOpcode(void) const { return Opcode; }
  void setOpcode(unsigned Opcode_) { Opcode = Opcode_; }

  unsigned getNewOpcode(void) const { return NewOpcode; }
  void setNewOpcode(unsigned NewOpcode_) { NewOpcode = NewOpcode_; }

  bool operator==(const WideningIntegerSolutionInfo &o) noexcept {
    bool MayEqual = Kind == o.getKind() && Opcode == o.getOpcode() &&
                    NewOpcode == o.getNewOpcode() && Cost == o.getCost() &&
                    Width == o.getWidth() &&
                    UpdatedWidth == o.getUpdatedWidth();
    if (MayEqual) {
      return this->getOperands() == o.getOperands();
    }
    return false;
  }

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
  inline int IsRedudant(WideningIntegerSolutionInfo &b) const noexcept;

}; // WideningIntegerSolutionInfo

class WIA_BINOP : public WideningIntegerSolutionInfo {
public:
  ~WIA_BINOP() {}
  WIA_BINOP() = delete;
  WIA_BINOP(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
            unsigned short FillTypeWidth_, unsigned short Width_,
            unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_BINOP, V_) {}
  WIA_BINOP(WIA_BINOP const &) = delete;
  WIA_BINOP &operator=(WIA_BINOP const &) = delete;

  static inline bool classof(WIA_BINOP const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_BINOP:
      return true;
    default:
      return false;
    }
  }
};

class WIA_PHI : public WideningIntegerSolutionInfo {
public:
  ~WIA_PHI() {}
  WIA_PHI() = delete;
  WIA_PHI(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
          unsigned short FillTypeWidth_, unsigned short Width_,
          unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_PHI, V_) {}
  WIA_PHI(WIA_PHI const &) = delete;
  WIA_PHI &operator=(WIA_PHI const &) = delete;

  static inline bool classof(WIA_PHI const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_PHI:
      return true;
    default:
      return false;
    }
  }
};

class WIA_FILL : public WideningIntegerSolutionInfo {
public:
  ~WIA_FILL() {};
  WIA_FILL() = delete;
  WIA_FILL(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
           unsigned short FillTypeWidth_, unsigned short Width_,
           unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_FILL, V_) {}
  WIA_FILL(WIA_FILL const &) = delete;
  WIA_FILL &operator=(WIA_FILL const &) = delete;

  static inline bool classof(WIA_FILL const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_FILL:
      return true;
    default:
      return false;
    }
  }
};

class WIA_WIDEN : public WideningIntegerSolutionInfo {
public:
  WIA_WIDEN() = delete;
  ~WIA_WIDEN() {}
  WIA_WIDEN(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
            unsigned short FillTypeWidth_, unsigned short Width_,
            unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_WIDEN, V_) {}
  WIA_WIDEN(WIA_WIDEN const &) = delete;
  WIA_WIDEN &operator=(WIA_WIDEN const &) = delete;

  static inline bool classof(WIA_WIDEN const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_WIDEN:
      return true;
    default:
      return false;
    }
  }
};

class WIA_WIDEN_GARBAGE : public WideningIntegerSolutionInfo {
public:
  WIA_WIDEN_GARBAGE() = delete;
  ~WIA_WIDEN_GARBAGE() {}
  WIA_WIDEN_GARBAGE(unsigned Opcode_, unsigned NewOpcode_,
                    IntegerFillType FillType_, unsigned short FillTypeWidth_,
                    unsigned short Width_, unsigned short UpdatedWidth_,
                    InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_WIDEN_GARBAGE, V_) {}
  WIA_WIDEN_GARBAGE(WIA_WIDEN_GARBAGE const &) = delete;
  WIA_WIDEN_GARBAGE &operator=(WIA_WIDEN_GARBAGE const &) = delete;

  static inline bool classof(WIA_WIDEN_GARBAGE const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_WIDEN_GARBAGE:
      return true;
    default:
      return false;
    }
  }
};

class WIA_NARROW : public WideningIntegerSolutionInfo {
public:
  WIA_NARROW() = delete;
  ~WIA_NARROW() {}
  WIA_NARROW(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
             unsigned short FillTypeWidth_, unsigned short Width_,
             unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_NARROW, V_) {}
  WIA_NARROW(WIA_NARROW const &) = delete;
  WIA_NARROW &operator=(WIA_NARROW const &) = delete;

  static inline bool classof(WIA_NARROW const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_NARROW:
      return true;
    default:
      return false;
    }
  }
};

class WIA_DROP_EXT : public WideningIntegerSolutionInfo {
public:
  WIA_DROP_EXT() = delete;
  ~WIA_DROP_EXT() {}
  WIA_DROP_EXT(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
               unsigned short FillTypeWidth_, unsigned short Width_,
               unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_DROP_EXT, V_) {}
  WIA_DROP_EXT(WIA_DROP_EXT const &) = delete;
  WIA_DROP_EXT &operator=(WIA_DROP_EXT const &) = delete;

  static inline bool classof(WIA_DROP_EXT const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_DROP_EXT:
      return true;
    default:
      return false;
    }
  }
};

class WIA_DROP_LOCOPY : public WideningIntegerSolutionInfo {
public:
  WIA_DROP_LOCOPY() = delete;
  ~WIA_DROP_LOCOPY() {}
  WIA_DROP_LOCOPY(unsigned Opcode_, unsigned NewOpcode_,
                  IntegerFillType FillType_, unsigned short FillTypeWidth_,
                  unsigned short Width_, unsigned short UpdatedWidth_,
                  InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_DROP_LOCOPY, V_) {}
  WIA_DROP_LOCOPY(WIA_DROP_LOCOPY const &) = delete;
  WIA_DROP_LOCOPY &operator=(WIA_DROP_LOCOPY const &) = delete;

  static inline bool classof(WIA_DROP_LOCOPY const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_DROP_LOCOPY:
      return true;
    default:
      return false;
    }
  }
};

class WIA_DROP_LOIGNORE : public WideningIntegerSolutionInfo {
public:
  WIA_DROP_LOIGNORE() = delete;
  ~WIA_DROP_LOIGNORE() {}
  WIA_DROP_LOIGNORE(unsigned Opcode_, unsigned NewOpcode_,
                    IntegerFillType FillType_, unsigned short FillTypeWidth_,
                    unsigned short Width_, unsigned short UpdatedWidth_,
                    InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_DROP_LOIGNORE, V_) {}
  WIA_DROP_LOIGNORE(WIA_DROP_LOIGNORE const &) = delete;
  WIA_DROP_LOIGNORE &operator=(WIA_DROP_LOIGNORE const &) = delete;

  static inline bool classof(WIA_DROP_LOIGNORE const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_DROP_LOIGNORE:
      return true;
    default:
      return false;
    }
  }
};

class WIA_EXTLO : public WideningIntegerSolutionInfo {
public:
  WIA_EXTLO() = delete;
  ~WIA_EXTLO() {}
  WIA_EXTLO(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
            unsigned short FillTypeWidth_, unsigned short Width_,
            unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_EXTLO, V_) {}
  WIA_EXTLO(WIA_EXTLO const &) = delete;
  WIA_EXTLO &operator=(WIA_EXTLO const &) = delete;

  static inline bool classof(WIA_EXTLO const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_EXTLO:
      return true;
    default:
      return false;
    }
  }
};

class WIA_SUBSUME_FILL : public WideningIntegerSolutionInfo {
public:
  WIA_SUBSUME_FILL() = delete;
  ~WIA_SUBSUME_FILL() {}
  WIA_SUBSUME_FILL(unsigned Opcode_, unsigned NewOpcode_,
                   IntegerFillType FillType_, unsigned short FillTypeWidth_,
                   unsigned short Width_, unsigned short UpdatedWidth_,
                   InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_SUBSUME_FILL, V_) {}
  WIA_SUBSUME_FILL(WIA_SUBSUME_FILL const &) = delete;
  WIA_SUBSUME_FILL &operator=(WIA_SUBSUME_FILL const &) = delete;

  static inline bool classof(WIA_SUBSUME_FILL const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_SUBSUME_FILL:
      return true;
    default:
      return false;
    }
  }
};

class WIA_SUBSUME_INDEX : public WideningIntegerSolutionInfo {
public:
  WIA_SUBSUME_INDEX() = delete;
  ~WIA_SUBSUME_INDEX() {}
  WIA_SUBSUME_INDEX(unsigned Opcode_, unsigned NewOpcode_,
                    IntegerFillType FillType_, unsigned short FillTypeWidth_,
                    unsigned short Width_, unsigned short UpdatedWidth_,
                    InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_SUBSUME_INDEX, V_) {}
  WIA_SUBSUME_INDEX(WIA_SUBSUME_INDEX const &) = delete;
  WIA_SUBSUME_INDEX &operator=(WIA_SUBSUME_INDEX const &) = delete;

  static inline bool classof(WIA_SUBSUME_INDEX const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_SUBSUME_INDEX:
      return true;
    default:
      return false;
    }
  }
};

class WIA_NATURAL : public WideningIntegerSolutionInfo {
public:
  WIA_NATURAL() = delete;
  ~WIA_NATURAL() {}
  WIA_NATURAL(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
              unsigned short FillTypeWidth_, unsigned short Width_,
              unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_NATURAL, V_) {}
  WIA_NATURAL(WIA_NATURAL const &) = delete;
  WIA_NATURAL &operator=(WIA_NATURAL const &) = delete;

  static inline bool classof(WIA_NATURAL const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {

    switch (Base->getKind()) {
    case WIAK_NATURAL:
      return true;
    default:
      return false;
    }
  }
};

class WIA_LOAD : public WideningIntegerSolutionInfo {
public:
  WIA_LOAD() = delete;
  ~WIA_LOAD() {}
  WIA_LOAD(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
           unsigned short FillTypeWidth_, unsigned short Width_,
           unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_LOAD, V_) {}
  WIA_LOAD(WIA_LOAD const &) = delete;
  WIA_LOAD &operator=(WIA_LOAD const &) = delete;

  static inline bool classof(WIA_LOAD const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_LOAD:
      return true;
    default:
      return false;
    }
  }
};

class WIA_STORE : public WideningIntegerSolutionInfo {
public:
  WIA_STORE() = delete;
  ~WIA_STORE() {}
  WIA_STORE(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
            unsigned short FillTypeWidth_, unsigned short Width_,
            unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_STORE, V_) {}
  WIA_STORE(WIA_STORE const &) = delete;
  WIA_STORE &operator=(WIA_STORE const &) = delete;

  static inline bool classof(WIA_STORE const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_STORE:
      return true;
    default:
      return false;
    }
  }
};

class WIA_UNOP : public WideningIntegerSolutionInfo {
public:
  WIA_UNOP() = delete;
  ~WIA_UNOP() {}
  WIA_UNOP(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
           unsigned short FillTypeWidth_, unsigned short Width_,
           unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_UNOP, V_) {}
  WIA_UNOP(WIA_UNOP const &) = delete;
  WIA_UNOP &operator=(WIA_UNOP const &) = delete;

  static inline bool classof(WIA_UNOP const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_UNOP:
      return true;
    default:
      return false;
    }
  }
};

class WIA_CONSTANT : public WideningIntegerSolutionInfo {
public:
  WIA_CONSTANT() = delete;
  ~WIA_CONSTANT() {}
  WIA_CONSTANT(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
               unsigned short FillTypeWidth_, unsigned short Width_,
               unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_CONSTANT, V_) {}
  WIA_CONSTANT(WIA_CONSTANT const &) = delete;
  WIA_CONSTANT &operator=(WIA_CONSTANT const &) = delete;

  static inline bool classof(WIA_CONSTANT const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_CONSTANT:
      return true;
    default:
      return false;
    }
  }
};

class WIA_NOOP : public WideningIntegerSolutionInfo {
public:
  WIA_NOOP() = delete;
  ~WIA_NOOP() {}
  WIA_NOOP(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
           unsigned short FillTypeWidth_, unsigned short Width_,
           unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_NOOP, V_) {}
  WIA_NOOP(WIA_NOOP const &) = delete;
  WIA_NOOP &operator=(WIA_NOOP const &) = delete;

  static inline bool classof(WIA_NOOP const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_NOOP:
      return true;
    default:
      return false;
    }
  }
};

class WIA_SWITCH : public WideningIntegerSolutionInfo {
public:
  WIA_SWITCH() = delete;
  ~WIA_SWITCH() {}
  WIA_SWITCH(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
             unsigned short FillTypeWidth_, unsigned short Width_,
             unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_SWITCH, V_) {}
  WIA_SWITCH(WIA_SWITCH const &) = delete;
  WIA_SWITCH &operator=(WIA_SWITCH const &) = delete;

  static inline bool classof(WIA_SWITCH const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_SWITCH:
      return true;
    default:
      return false;
    }
  }
};

class WIA_UNKNOWN : public WideningIntegerSolutionInfo {
public:
  WIA_UNKNOWN() = delete;
  ~WIA_UNKNOWN() {}
  WIA_UNKNOWN(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
              unsigned short FillTypeWidth_, unsigned short Width_,
              unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_UNKNOWN, V_) {}
  WIA_UNKNOWN(WIA_UNKNOWN const &) = delete;
  WIA_UNKNOWN &operator=(WIA_UNKNOWN const &) = delete;

  static inline bool classof(WIA_UNKNOWN const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_UNKNOWN:
      return true;
    default:
      return false;
    }
  }
};

class WIA_RET : public WideningIntegerSolutionInfo {
public:
  WIA_RET() = delete;
  ~WIA_RET() {}
  WIA_RET(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
          unsigned short FillTypeWidth_, unsigned short Width_,
          unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_RET, V_) {}
  WIA_RET(WIA_RET const &) = delete;
  WIA_RET &operator=(WIA_RET const &) = delete;

  static inline bool classof(WIA_RET const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_RET:
      return true;
    default:
      return false;
    }
  }
};

class WIA_SELECT : public WideningIntegerSolutionInfo {
public:
  WIA_SELECT() = delete;
  ~WIA_SELECT() {}
  WIA_SELECT(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
             unsigned short FillTypeWidth_, unsigned short Width_,
             unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_SELECT, V_) {}
  WIA_SELECT(WIA_SELECT const &) = delete;
  WIA_SELECT &operator=(WIA_SELECT const &) = delete;

  static inline bool classof(WIA_SELECT const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_SELECT:
      return true;
    default:
      return false;
    }
  }
};

class WIA_ICMP : public WideningIntegerSolutionInfo {
public:
  WIA_ICMP() = delete;
  ~WIA_ICMP() {}
  WIA_ICMP(unsigned Opcode_, unsigned NewOpcode_, IntegerFillType FillType_,
           unsigned short FillTypeWidth_, unsigned short Width_,
           unsigned short UpdatedWidth_, InstructionCost Cost_, Value *V_)
      : WideningIntegerSolutionInfo::WideningIntegerSolutionInfo(
            Opcode_, NewOpcode_, FillType_, FillTypeWidth_, Width_,
            UpdatedWidth_, Cost_, WIAK_ICMP, V_) {}
  WIA_ICMP(WIA_ICMP const &) = delete;
  WIA_ICMP &operator=(WIA_ICMP const &) = delete;

  static inline bool classof(WIA_ICMP const *) { return true; }
  static inline bool classof(WideningIntegerSolutionInfo const *Base) {
    switch (Base->getKind()) {
    case WIAK_ICMP:
      return true;
    default:
      return false;
    }
  }
  unsigned short getUpdatedIcmpWidth(void) const { return UpdatedIcmpWidth; }

  void setUpdatedIcmpWidth(unsigned short UpdatedIcmpWidth_) {
    UpdatedIcmpWidth = UpdatedIcmpWidth_;
  }

private:
  unsigned short UpdatedIcmpWidth;
};

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
  if (auto *IcmpSol = dyn_cast<WIA_ICMP>(&Sol)) {
    out << "\tUpdatedIcmpWidth : " << IcmpSol->getUpdatedIcmpWidth() << '\n';
  }

  return out;
}

// Returns 0 if no one is redudant
// Returns -1 if b is redundant given solution this
// Returns 1 if "this" is redudant given solution b
inline int WideningIntegerSolutionInfo::IsRedudant(
    WideningIntegerSolutionInfo &b) const noexcept {

  int n1 = FillTypeWidth;
  auto c1 = Cost;
  int n2 = b.getFillTypeWidth();
  auto c2 = b.getCost();
  int w1 = UpdatedWidth;
  int w2 = b.getUpdatedWidth();
  int FillType1 = FillType;
  int FillType2 = b.getFillType();

  if (auto *bIcmp = dyn_cast<WIA_ICMP>(&b)) {
    w2 = bIcmp->getUpdatedIcmpWidth();
  }
  if (auto *thisIcmp = dyn_cast<WIA_ICMP>(this)) {
    w1 = thisIcmp->getUpdatedIcmpWidth();
  }
  // different updatedWidth or different FillType or different Inst no
  // redundant.
  if (w1 != w2 || FillType1 != FillType2 || b.getValue() != this->getValue()) {
    return 0;
  }

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
