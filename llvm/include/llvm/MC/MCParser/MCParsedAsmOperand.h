//===-- llvm/MC/MCParsedAsmOperand.h - Asm Parser Operand -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMOPERAND_H
#define LLVM_MC_MCASMOPERAND_H

namespace llvm {
class SMLoc;
class raw_ostream;

/// MCParsedAsmOperand - This abstract class represents a source-level assembly
/// instruction operand.  It should be subclassed by target-specific code.  This
/// base class is used by target-independent clients and is the interface
/// between parsing an asm instruction and recognizing it.
class MCParsedAsmOperand {
  /// MCOperandNum - The corresponding MCInst operand number.  Only valid when
  /// parsing MS-style inline assembly.
  unsigned MCOperandNum;

  /// Constraint - The constraint on this operand.  Only valid when parsing
  /// MS-style inline assembly.
  std::string Constraint;

public:
  MCParsedAsmOperand() {}
  virtual ~MCParsedAsmOperand() {}

  void setConstraint(StringRef C) { Constraint = C.str(); }
  StringRef getConstraint() { return Constraint; }

  void setMCOperandNum (unsigned OpNum) { MCOperandNum = OpNum; }
  unsigned getMCOperandNum() { return MCOperandNum; }

  unsigned getNameLen() {
    assert (getStartLoc().isValid() && "Invalid StartLoc!");
    assert (getEndLoc().isValid() && "Invalid EndLoc!");
    return getEndLoc().getPointer() - getStartLoc().getPointer();
  }

  StringRef getName() {
    return StringRef(getStartLoc().getPointer(), getNameLen());
  }

  /// isToken - Is this a token operand?
  virtual bool isToken() const = 0;
  /// isImm - Is this an immediate operand?
  virtual bool isImm() const = 0;
  /// isReg - Is this a register operand?
  virtual bool isReg() const = 0;
  virtual unsigned getReg() const = 0;

  /// isMem - Is this a memory operand?
  virtual bool isMem() const = 0;
  virtual unsigned getMemSize() const { return 0; }

  /// getStartLoc - Get the location of the first token of this operand.
  virtual SMLoc getStartLoc() const = 0;
  /// getEndLoc - Get the location of the last token of this operand.
  virtual SMLoc getEndLoc() const = 0;

  /// needAsmRewrite - AsmRewrites happen in both the target-independent and
  /// target-dependent parsers.  The target-independent parser calls this
  /// function to determine if the target-dependent parser has already taken
  /// care of the rewrites.  Only valid when parsing MS-style inline assembly.
  virtual bool needAsmRewrite() const { return true; }

  /// isOffsetOf - Do we need to emit code to get the offset of the variable,
  /// rather then the value of the variable?   Only valid when parsing MS-style
  /// inline assembly.
  virtual bool isOffsetOf() const { return false; }

  /// getOffsetOfLoc - Get the location of the offset operator.
  virtual SMLoc getOffsetOfLoc() const { return SMLoc(); }

  /// needSizeDirective - Do we need to emit a sizing directive for this
  /// operand?  Only valid when parsing MS-style inline assembly.
  virtual bool needSizeDirective() const { return false; }

  /// print - Print a debug representation of the operand to the given stream.
  virtual void print(raw_ostream &OS) const = 0;
  /// dump - Print to the debug stream.
  virtual void dump() const;
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline raw_ostream& operator<<(raw_ostream &OS, const MCParsedAsmOperand &MO) {
  MO.print(OS);
  return OS;
}

} // end namespace llvm.

#endif
