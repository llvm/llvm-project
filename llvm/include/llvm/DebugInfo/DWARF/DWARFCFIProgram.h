//===- DWARFCFIProgram.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFCFIPROGRAM_H
#define LLVM_DEBUGINFO_DWARF_DWARFCFIPROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <map>
#include <memory>
#include <vector>

namespace llvm {

namespace dwarf {
/// Represent a sequence of Call Frame Information instructions that, when read
/// in order, construct a table mapping PC to frame state. This can also be
/// referred to as "CFI rules" in DWARF literature to avoid confusion with
/// computer programs in the broader sense, and in this context each instruction
/// would be a rule to establish the mapping. Refer to pg. 172 in the DWARF5
/// manual, "6.4.1 Structure of Call Frame Information".
class CFIProgram {
public:
  static constexpr size_t MaxOperands = 3;
  typedef SmallVector<uint64_t, MaxOperands> Operands;

  /// An instruction consists of a DWARF CFI opcode and an optional sequence of
  /// operands. If it refers to an expression, then this expression has its own
  /// sequence of operations and operands handled separately by DWARFExpression.
  struct Instruction {
    Instruction(uint8_t Opcode) : Opcode(Opcode) {}

    uint8_t Opcode;
    Operands Ops;
    // Associated DWARF expression in case this instruction refers to one
    std::optional<DWARFExpression> Expression;

    Expected<uint64_t> getOperandAsUnsigned(const CFIProgram &CFIP,
                                            uint32_t OperandIdx) const;

    Expected<int64_t> getOperandAsSigned(const CFIProgram &CFIP,
                                         uint32_t OperandIdx) const;
  };

  using InstrList = std::vector<Instruction>;
  using iterator = InstrList::iterator;
  using const_iterator = InstrList::const_iterator;

  iterator begin() { return Instructions.begin(); }
  const_iterator begin() const { return Instructions.begin(); }
  iterator end() { return Instructions.end(); }
  const_iterator end() const { return Instructions.end(); }

  unsigned size() const { return (unsigned)Instructions.size(); }
  bool empty() const { return Instructions.empty(); }
  uint64_t codeAlign() const { return CodeAlignmentFactor; }
  int64_t dataAlign() const { return DataAlignmentFactor; }
  Triple::ArchType triple() const { return Arch; }

  CFIProgram(uint64_t CodeAlignmentFactor, int64_t DataAlignmentFactor,
             Triple::ArchType Arch)
      : CodeAlignmentFactor(CodeAlignmentFactor),
        DataAlignmentFactor(DataAlignmentFactor), Arch(Arch) {}

  /// Parse and store a sequence of CFI instructions from Data,
  /// starting at *Offset and ending at EndOffset. *Offset is updated
  /// to EndOffset upon successful parsing, or indicates the offset
  /// where a problem occurred in case an error is returned.
  Error parse(DWARFDataExtractor Data, uint64_t *Offset, uint64_t EndOffset);

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts, unsigned IndentLevel,
            std::optional<uint64_t> InitialLocation) const;

  void addInstruction(const Instruction &I) { Instructions.push_back(I); }

  /// Get a DWARF CFI call frame string for the given DW_CFA opcode.
  StringRef callFrameString(unsigned Opcode) const;

private:
  std::vector<Instruction> Instructions;
  const uint64_t CodeAlignmentFactor;
  const int64_t DataAlignmentFactor;
  Triple::ArchType Arch;

  /// Convenience method to add a new instruction with the given opcode.
  void addInstruction(uint8_t Opcode) {
    Instructions.push_back(Instruction(Opcode));
  }

  /// Add a new single-operand instruction.
  void addInstruction(uint8_t Opcode, uint64_t Operand1) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
  }

  /// Add a new instruction that has two operands.
  void addInstruction(uint8_t Opcode, uint64_t Operand1, uint64_t Operand2) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
    Instructions.back().Ops.push_back(Operand2);
  }

  /// Add a new instruction that has three operands.
  void addInstruction(uint8_t Opcode, uint64_t Operand1, uint64_t Operand2,
                      uint64_t Operand3) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
    Instructions.back().Ops.push_back(Operand2);
    Instructions.back().Ops.push_back(Operand3);
  }

  /// Types of operands to CFI instructions
  /// In DWARF, this type is implicitly tied to a CFI instruction opcode and
  /// thus this type doesn't need to be explicitly written to the file (this is
  /// not a DWARF encoding). The relationship of instrs to operand types can
  /// be obtained from getOperandTypes() and is only used to simplify
  /// instruction printing.
  enum OperandType {
    OT_Unset,
    OT_None,
    OT_Address,
    OT_Offset,
    OT_FactoredCodeOffset,
    OT_SignedFactDataOffset,
    OT_UnsignedFactDataOffset,
    OT_Register,
    OT_AddressSpace,
    OT_Expression
  };

  /// Get the OperandType as a "const char *".
  static const char *operandTypeString(OperandType OT);

  /// Retrieve the array describing the types of operands according to the enum
  /// above. This is indexed by opcode.
  static ArrayRef<OperandType[MaxOperands]> getOperandTypes();

  /// Print \p Opcode's operand number \p OperandIdx which has value \p Operand.
  void printOperand(raw_ostream &OS, DIDumpOptions DumpOpts,
                    const Instruction &Instr, unsigned OperandIdx,
                    uint64_t Operand, std::optional<uint64_t> &Address) const;
};

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCFIPROGRAM_H
