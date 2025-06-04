//===--- DWARFExpression.h - DWARF Expression handling ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFEXPRESSION_H
#define LLVM_DEBUGINFO_DWARF_DWARFEXPRESSION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {
class DWARFUnit;
struct DIDumpOptions;
class MCRegisterInfo;
class raw_ostream;

class DWARFExpression {
public:
  class iterator;

  /// This class represents an Operation in the Expression.
  ///
  /// An Operation can be in Error state (check with isError()). This
  /// means that it couldn't be decoded successfully and if it is the
  /// case, all others fields contain undefined values.
  class Operation {
  public:
    /// Size and signedness of expression operations' operands.
    enum Encoding : uint8_t {
      Size1 = 0,
      Size2 = 1,
      Size4 = 2,
      Size8 = 3,
      SizeLEB = 4,
      SizeAddr = 5,
      SizeRefAddr = 6,
      SizeBlock = 7, ///< Preceding operand contains block size
      BaseTypeRef = 8,
      /// The operand is a ULEB128 encoded SubOpcode. This is only valid
      /// for the first operand of an operation.
      SizeSubOpLEB = 9,
      WasmLocationArg = 30,
      SignBit = 0x80,
      SignedSize1 = SignBit | Size1,
      SignedSize2 = SignBit | Size2,
      SignedSize4 = SignBit | Size4,
      SignedSize8 = SignBit | Size8,
      SignedSizeLEB = SignBit | SizeLEB,
    };

    enum DwarfVersion : uint8_t {
      DwarfNA, ///< Serves as a marker for unused entries
      Dwarf2 = 2,
      Dwarf3,
      Dwarf4,
      Dwarf5
    };

    /// Description of the encoding of one expression Op.
    struct Description {
      DwarfVersion Version; ///< Dwarf version where the Op was introduced.
      SmallVector<Encoding> Op; ///< Encoding for Op operands.

      template <typename... Ts>
      Description(DwarfVersion Version, Ts... Op)
          : Version(Version), Op{Op...} {}
      Description() : Description(DwarfNA) {}
      ~Description() = default;
    };

  private:
    friend class DWARFExpression::iterator;
    friend class DWARFExpressionPrinter;
    friend class DWARFVerifier;

    uint8_t Opcode; ///< The Op Opcode, DW_OP_<something>.
    Description Desc;
    bool Error = false;
    uint64_t EndOffset;
    SmallVector<uint64_t> Operands;
    SmallVector<uint64_t> OperandEndOffsets;

  public:
    const Description &getDescription() const { return Desc; }
    uint8_t getCode() const { return Opcode; }
    std::optional<unsigned> getSubCode() const;
    uint64_t getNumOperands() const { return Operands.size(); }
    ArrayRef<uint64_t> getRawOperands() const { return Operands; };
    uint64_t getRawOperand(unsigned Idx) const { return Operands[Idx]; }
    ArrayRef<uint64_t> getOperandEndOffsets() const {
      return OperandEndOffsets;
    }
    uint64_t getOperandEndOffset(unsigned Idx) const {
      return OperandEndOffsets[Idx];
    }
    uint64_t getEndOffset() const { return EndOffset; }
    bool isError() const { return Error; }

  private:
    bool extract(DataExtractor Data, uint8_t AddressSize, uint64_t Offset,
                 std::optional<dwarf::DwarfFormat> Format);
  };

  /// An iterator to go through the expression operations.
  class iterator
      : public iterator_facade_base<iterator, std::forward_iterator_tag,
                                    const Operation> {
    friend class DWARFExpression;
    const DWARFExpression *Expr;
    uint64_t Offset;
    Operation Op;
    iterator(const DWARFExpression *Expr, uint64_t Offset)
        : Expr(Expr), Offset(Offset) {
      Op.Error =
          Offset >= Expr->Data.getData().size() ||
          !Op.extract(Expr->Data, Expr->AddressSize, Offset, Expr->Format);
    }

  public:
    iterator &operator++() {
      Offset = Op.isError() ? Expr->Data.getData().size() : Op.EndOffset;
      Op.Error =
          Offset >= Expr->Data.getData().size() ||
          !Op.extract(Expr->Data, Expr->AddressSize, Offset, Expr->Format);
      return *this;
    }

    const Operation &operator*() const { return Op; }

    iterator skipBytes(uint64_t Add) const {
      return iterator(Expr, Op.EndOffset + Add);
    }

    // Comparison operators are provided out of line.
    friend bool operator==(const iterator &, const iterator &);
  };

  DWARFExpression(DataExtractor Data, uint8_t AddressSize,
                  std::optional<dwarf::DwarfFormat> Format = std::nullopt)
      : Data(Data), AddressSize(AddressSize), Format(Format) {
    assert(AddressSize == 8 || AddressSize == 4 || AddressSize == 2);
  }

  iterator begin() const { return iterator(this, 0); }
  iterator end() const { return iterator(this, Data.getData().size()); }

  bool operator==(const DWARFExpression &RHS) const;

  StringRef getData() const { return Data.getData(); }

  friend class DWARFExpressionPrinter;
  friend class DWARFVerifier;

private:
  DataExtractor Data;
  uint8_t AddressSize;
  std::optional<dwarf::DwarfFormat> Format;
};

inline bool operator==(const DWARFExpression::iterator &LHS,
                       const DWARFExpression::iterator &RHS) {
  return LHS.Expr == RHS.Expr && LHS.Offset == RHS.Offset;
}

// This functionality is separated from the main data structure so that nothing
// in DWARFExpression.cpp needs build-time dependencies on DWARFUnit or other
// higher-level Dwarf structures. This approach creates better layering and
// allows DWARFExpression to be used from code which can't have dependencies on
// those higher-level structures.

class DWARFUnit;
struct DIDumpOptions;
class raw_ostream;

class DWARFExpressionPrinter {
public:
  /// Print a Dwarf expression/
  /// \param E to be printed
  /// \param OS to this stream
  /// \param GetNameForDWARFReg callback to return dwarf register name
  static void print(const DWARFExpression *E, raw_ostream &OS,
                    DIDumpOptions DumpOpts, DWARFUnit *U, bool IsEH = false);

  /// Print the expression in a format intended to be compact and useful to a
  /// user, but not perfectly unambiguous, or capable of representing every
  /// valid DWARF expression. Returns true if the expression was sucessfully
  /// printed.
  ///
  /// \param E to be printed
  /// \param OS to this stream
  /// \param GetNameForDWARFReg callback to return dwarf register name
  ///
  /// \returns true if the expression was successfully printed
  static bool printCompact(const DWARFExpression *E, raw_ostream &OS,
                           std::function<StringRef(uint64_t RegNum, bool IsEH)>
                               GetNameForDWARFReg = nullptr);

  /// Pretty print a register opcode and operands.
  /// \param U within the context of this Dwarf unit, if any.
  /// \param OS to this stream
  /// \param DumpOpts with these options
  /// \param Opcode to print
  /// \param Operands to the opcode
  ///
  /// returns true if the Op was successfully printed
  static bool prettyPrintRegisterOp(DWARFUnit *U, raw_ostream &OS,
                                    DIDumpOptions DumpOpts, uint8_t Opcode,
                                    ArrayRef<uint64_t> Operands);

private:
  static bool printOp(const DWARFExpression::Operation *Op, raw_ostream &OS,
                      DIDumpOptions DumpOpts, const DWARFExpression *Expr,
                      DWARFUnit *U);

  static void prettyPrintBaseTypeRef(DWARFUnit *U, raw_ostream &OS,
                                     DIDumpOptions DumpOpts,
                                     ArrayRef<uint64_t> Operands,
                                     unsigned Operand);
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFEXPRESSION_H
