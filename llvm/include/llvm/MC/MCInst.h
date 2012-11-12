//===-- llvm/MC/MCInst.h - MCInst class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCInst and MCOperand classes, which
// is the basic representation used to represent low-level machine code
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINST_H
#define LLVM_MC_MCINST_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class raw_ostream;
class MCAsmInfo;
class MCInstPrinter;
class MCExpr;
class MCInst;

/// MCOperand - Instances of this class represent operands of the MCInst class.
/// This is a simple discriminated union.
class MCOperand {
  enum MachineOperandType {
    kInvalid,                 ///< Uninitialized.
    kRegister,                ///< Register operand.
    kImmediate,               ///< Immediate operand.
    kFPImmediate,             ///< Floating-point immediate operand.
    kExpr,                    ///< Relocatable immediate operand.
    kInst                     ///< Sub-instruction operand.
  };
  unsigned char Kind;

  union {
    unsigned RegVal;
    int64_t ImmVal;
    double FPImmVal;
    const MCExpr *ExprVal;
    const MCInst *InstVal;
  };
public:

  MCOperand() : Kind(kInvalid), FPImmVal(0.0) {}

  bool isValid() const { return Kind != kInvalid; }
  bool isReg() const { return Kind == kRegister; }
  bool isImm() const { return Kind == kImmediate; }
  bool isFPImm() const { return Kind == kFPImmediate; }
  bool isExpr() const { return Kind == kExpr; }
  bool isInst() const { return Kind == kInst; }

  /// getReg - Returns the register number.
  unsigned getReg() const {
    assert(isReg() && "This is not a register operand!");
    return RegVal;
  }

  /// setReg - Set the register number.
  void setReg(unsigned Reg) {
    assert(isReg() && "This is not a register operand!");
    RegVal = Reg;
  }

  int64_t getImm() const {
    assert(isImm() && "This is not an immediate");
    return ImmVal;
  }
  void setImm(int64_t Val) {
    assert(isImm() && "This is not an immediate");
    ImmVal = Val;
  }

  double getFPImm() const {
    assert(isFPImm() && "This is not an FP immediate");
    return FPImmVal;
  }

  void setFPImm(double Val) {
    assert(isFPImm() && "This is not an FP immediate");
    FPImmVal = Val;
  }

  const MCExpr *getExpr() const {
    assert(isExpr() && "This is not an expression");
    return ExprVal;
  }
  void setExpr(const MCExpr *Val) {
    assert(isExpr() && "This is not an expression");
    ExprVal = Val;
  }

  const MCInst *getInst() const {
    assert(isInst() && "This is not a sub-instruction");
    return InstVal;
  }
  void setInst(const MCInst *Val) {
    assert(isInst() && "This is not a sub-instruction");
    InstVal = Val;
  }

  static MCOperand CreateReg(unsigned Reg) {
    MCOperand Op;
    Op.Kind = kRegister;
    Op.RegVal = Reg;
    return Op;
  }
  static MCOperand CreateImm(int64_t Val) {
    MCOperand Op;
    Op.Kind = kImmediate;
    Op.ImmVal = Val;
    return Op;
  }
  static MCOperand CreateFPImm(double Val) {
    MCOperand Op;
    Op.Kind = kFPImmediate;
    Op.FPImmVal = Val;
    return Op;
  }
  static MCOperand CreateExpr(const MCExpr *Val) {
    MCOperand Op;
    Op.Kind = kExpr;
    Op.ExprVal = Val;
    return Op;
  }
  static MCOperand CreateInst(const MCInst *Val) {
    MCOperand Op;
    Op.Kind = kInst;
    Op.InstVal = Val;
    return Op;
  }

  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;
  void dump() const;
};

template <> struct isPodLike<MCOperand> { static const bool value = true; };

/// MCInst - Instances of this class represent a single low-level machine
/// instruction.
class MCInst {
  unsigned Opcode;
  SMLoc Loc;
  SmallVector<MCOperand, 8> Operands;
public:
  MCInst() : Opcode(0) {}

  void setOpcode(unsigned Op) { Opcode = Op; }
  unsigned getOpcode() const { return Opcode; }

  void setLoc(SMLoc loc) { Loc = loc; }
  SMLoc getLoc() const { return Loc; }

  const MCOperand &getOperand(unsigned i) const { return Operands[i]; }
  MCOperand &getOperand(unsigned i) { return Operands[i]; }
  unsigned getNumOperands() const { return Operands.size(); }

  void addOperand(const MCOperand &Op) {
    Operands.push_back(Op);
  }

  void clear() { Operands.clear(); }
  size_t size() { return Operands.size(); }

  typedef SmallVector<MCOperand, 8>::iterator iterator;
  iterator begin() { return Operands.begin(); }
  iterator end()   { return Operands.end();   }
  iterator insert(iterator I, const MCOperand &Op) {
    return Operands.insert(I, Op);
  }

  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;
  void dump() const;

  /// \brief Dump the MCInst as prettily as possible using the additional MC
  /// structures, if given. Operators are separated by the \p Separator
  /// string.
  void dump_pretty(raw_ostream &OS, const MCAsmInfo *MAI = 0,
                   const MCInstPrinter *Printer = 0,
                   StringRef Separator = " ") const;
};

inline raw_ostream& operator<<(raw_ostream &OS, const MCOperand &MO) {
  MO.print(OS, 0);
  return OS;
}

inline raw_ostream& operator<<(raw_ostream &OS, const MCInst &MI) {
  MI.print(OS, 0);
  return OS;
}

} // end namespace llvm

#endif
