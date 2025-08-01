//===- DWARFCFIProgram.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFCFIPROGRAM_H
#define LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFCFIPROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFDataExtractorSimple.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/Support/Compiler.h"
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

    LLVM_ABI Expected<uint64_t> getOperandAsUnsigned(const CFIProgram &CFIP,
                                                     uint32_t OperandIdx) const;

    LLVM_ABI Expected<int64_t> getOperandAsSigned(const CFIProgram &CFIP,
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
  template <typename T>
  Error parse(DWARFDataExtractorBase<T> &Data, uint64_t *Offset,
              uint64_t EndOffset) {
    // See DWARF standard v3, section 7.23
    const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
    const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;

    DataExtractor::Cursor C(*Offset);
    while (C && C.tell() < EndOffset) {
      uint8_t Opcode = Data.getRelocatedValue(C, 1);
      if (!C)
        break;

      // Some instructions have a primary opcode encoded in the top bits.
      if (uint8_t Primary = Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK) {
        // If it's a primary opcode, the first operand is encoded in the
        // bottom bits of the opcode itself.
        uint64_t Op1 = Opcode & DWARF_CFI_PRIMARY_OPERAND_MASK;
        switch (Primary) {
        case DW_CFA_advance_loc:
        case DW_CFA_restore:
          addInstruction(Primary, Op1);
          break;
        case DW_CFA_offset:
          addInstruction(Primary, Op1, Data.getULEB128(C));
          break;
        default:
          llvm_unreachable("invalid primary CFI opcode");
        }
        continue;
      }

      // Extended opcode - its value is Opcode itself.
      switch (Opcode) {
      default:
        return createStringError(errc::illegal_byte_sequence,
                                 "invalid extended CFI opcode 0x%" PRIx8,
                                 Opcode);
      case DW_CFA_nop:
      case DW_CFA_remember_state:
      case DW_CFA_restore_state:
      case DW_CFA_GNU_window_save:
      case DW_CFA_AARCH64_negate_ra_state_with_pc:
        // No operands
        addInstruction(Opcode);
        break;
      case DW_CFA_set_loc:
        // Operands: Address
        addInstruction(Opcode, Data.getRelocatedAddress(C));
        break;
      case DW_CFA_advance_loc1:
        // Operands: 1-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(C, 1));
        break;
      case DW_CFA_advance_loc2:
        // Operands: 2-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(C, 2));
        break;
      case DW_CFA_advance_loc4:
        // Operands: 4-byte delta
        addInstruction(Opcode, Data.getRelocatedValue(C, 4));
        break;
      case DW_CFA_restore_extended:
      case DW_CFA_undefined:
      case DW_CFA_same_value:
      case DW_CFA_def_cfa_register:
      case DW_CFA_def_cfa_offset:
      case DW_CFA_GNU_args_size:
        // Operands: ULEB128
        addInstruction(Opcode, Data.getULEB128(C));
        break;
      case DW_CFA_def_cfa_offset_sf:
        // Operands: SLEB128
        addInstruction(Opcode, Data.getSLEB128(C));
        break;
      case DW_CFA_LLVM_def_aspace_cfa:
      case DW_CFA_LLVM_def_aspace_cfa_sf: {
        auto RegNum = Data.getULEB128(C);
        auto CfaOffset = Opcode == DW_CFA_LLVM_def_aspace_cfa
                             ? Data.getULEB128(C)
                             : Data.getSLEB128(C);
        auto AddressSpace = Data.getULEB128(C);
        addInstruction(Opcode, RegNum, CfaOffset, AddressSpace);
        break;
      }
      case DW_CFA_offset_extended:
      case DW_CFA_register:
      case DW_CFA_def_cfa:
      case DW_CFA_val_offset: {
        // Operands: ULEB128, ULEB128
        // Note: We can not embed getULEB128 directly into function
        // argument list. getULEB128 changes Offset and order of evaluation
        // for arguments is unspecified.
        uint64_t op1 = Data.getULEB128(C);
        uint64_t op2 = Data.getULEB128(C);
        addInstruction(Opcode, op1, op2);
        break;
      }
      case DW_CFA_offset_extended_sf:
      case DW_CFA_def_cfa_sf:
      case DW_CFA_val_offset_sf: {
        // Operands: ULEB128, SLEB128
        // Note: see comment for the previous case
        uint64_t op1 = Data.getULEB128(C);
        uint64_t op2 = (uint64_t)Data.getSLEB128(C);
        addInstruction(Opcode, op1, op2);
        break;
      }
      case DW_CFA_def_cfa_expression: {
        uint64_t ExprLength = Data.getULEB128(C);
        addInstruction(Opcode, 0);
        StringRef Expression = Data.getBytes(C, ExprLength);

        DataExtractor Extractor(Expression, Data.isLittleEndian(),
                                Data.getAddressSize());
        // Note. We do not pass the DWARF format to DWARFExpression, because
        // DW_OP_call_ref, the only operation which depends on the format, is
        // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
        Instructions.back().Expression =
            DWARFExpression(Extractor, Data.getAddressSize());
        break;
      }
      case DW_CFA_expression:
      case DW_CFA_val_expression: {
        uint64_t RegNum = Data.getULEB128(C);
        addInstruction(Opcode, RegNum, 0);

        uint64_t BlockLength = Data.getULEB128(C);
        StringRef Expression = Data.getBytes(C, BlockLength);
        DataExtractor Extractor(Expression, Data.isLittleEndian(),
                                Data.getAddressSize());
        // Note. We do not pass the DWARF format to DWARFExpression, because
        // DW_OP_call_ref, the only operation which depends on the format, is
        // prohibited in call frame instructions, see sec. 6.4.2 in DWARFv5.
        Instructions.back().Expression =
            DWARFExpression(Extractor, Data.getAddressSize());
        break;
      }
      }
    }

    *Offset = C.tell();
    return C.takeError();
  }

  void addInstruction(const Instruction &I) { Instructions.push_back(I); }

  /// Get a DWARF CFI call frame string for the given DW_CFA opcode.
  LLVM_ABI StringRef callFrameString(unsigned Opcode) const;

  /// Types of operands to CFI instructions
  /// In DWARF, this type is implicitly tied to a CFI instruction opcode and
  /// thus this type doesn't need to be explicitly written to the file (this is
  /// not a DWARF encoding). The relationship of instrs to operand types can
  /// be obtained from getOperandTypes() and is only used to simplify
  /// instruction printing and error messages.
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
  LLVM_ABI static const char *operandTypeString(OperandType OT);

  /// Retrieve the array describing the types of operands according to the enum
  /// above. This is indexed by opcode.
  LLVM_ABI static ArrayRef<OperandType[MaxOperands]> getOperandTypes();

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

private:
  std::vector<Instruction> Instructions;
  const uint64_t CodeAlignmentFactor;
  const int64_t DataAlignmentFactor;
  Triple::ArchType Arch;
};

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_LOWLEVEL_DWARFCFIPROGRAM_H
