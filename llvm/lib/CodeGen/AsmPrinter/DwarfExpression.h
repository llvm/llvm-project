//===- llvm/CodeGen/DwarfExpression.h - Dwarf Compile Unit ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H

#include "ByteStreamer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <cassert>
#include <cstdint>
#include <iterator>

namespace llvm {

class AsmPrinter;
class APInt;
class DwarfCompileUnit;
class DIELoc;
class TargetRegisterInfo;

/// Holds a DIExpression and keeps track of how many operands have been consumed
/// so far.
class DIExpressionCursor {
  DIExpression::expr_op_iterator Start, End;

public:
  DIExpressionCursor(const DIExpression *Expr) {
    if (!Expr) {
      assert(Start == End);
      return;
    }
    Start = Expr->expr_op_begin();
    End = Expr->expr_op_end();
  }

  DIExpressionCursor(ArrayRef<uint64_t> Expr)
      : Start(Expr.begin()), End(Expr.end()) {}

  DIExpressionCursor(const DIExpressionCursor &) = default;

  /// Consume one operation.
  Optional<DIExpression::ExprOperand> take() {
    if (Start == End)
      return None;
    return *(Start++);
  }

  /// Consume N operations.
  void consume(unsigned N) { std::advance(Start, N); }

  /// Return the current operation.
  Optional<DIExpression::ExprOperand> peek() const {
    if (Start == End)
      return None;
    return *(Start);
  }

  /// Return the next operation.
  Optional<DIExpression::ExprOperand> peekNext() const {
    if (Start == End)
      return None;

    auto Next = Start.getNext();
    if (Next == End)
      return None;

    return *Next;
  }

  /// Determine whether there are any operations left in this expression.
  operator bool() const { return Start != End; }

  DIExpression::expr_op_iterator begin() const { return Start; }
  DIExpression::expr_op_iterator end() const { return End; }

  /// Retrieve the fragment information, if any.
  Optional<DIExpression::FragmentInfo> getFragmentInfo() const {
    return DIExpression::getFragmentInfo(Start, End);
  }
};

/// Base class containing the logic for constructing DWARF expressions
/// independently of whether they are emitted into a DIE or into a .debug_loc
/// entry.
///
/// Some DWARF operations, e.g. DW_OP_entry_value, need to calculate the size
/// of a succeeding DWARF block before the latter is emitted to the output.
/// To handle such cases, data can conditionally be emitted to a temporary
/// buffer, which can later on be committed to the main output. The size of the
/// temporary buffer is queryable, allowing for the size of the data to be
/// emitted before the data is committed.
class DwarfExpression {
protected:
  /// Holds information about all subregisters comprising a register location.
  struct Register {
    int DwarfRegNo;
    unsigned Size;
    const char *Comment;
  };

  /// Whether we are currently emitting an entry value operation.
  bool IsEmittingEntryValue = false;

  DwarfCompileUnit &CU;

  /// The register location, if any.
  SmallVector<Register, 2> DwarfRegs;

  /// Current Fragment Offset in Bits.
  uint64_t OffsetInBits = 0;

  /// Sometimes we need to add a DW_OP_bit_piece to describe a subregister.
  unsigned SubRegisterSizeInBits : 16;
  unsigned SubRegisterOffsetInBits : 16;

  /// The kind of location description being produced.
  enum { Unknown = 0, Register, Memory, Implicit };

  /// The flags of location description being produced.
  enum { EntryValue = 1, CallSiteParamValue };

  unsigned LocationKind : 3;
  unsigned LocationFlags : 2;
  unsigned DwarfVersion : 4;

public:
  bool isUnknownLocation() const {
    return LocationKind == Unknown;
  }

  bool isMemoryLocation() const {
    return LocationKind == Memory;
  }

  bool isRegisterLocation() const {
    return LocationKind == Register;
  }

  bool isImplicitLocation() const {
    return LocationKind == Implicit;
  }

  bool isEntryValue() const {
    return LocationFlags & EntryValue;
  }

  bool isParameterValue() {
    return LocationFlags & CallSiteParamValue;
  }

  Optional<uint8_t> TagOffset;

protected:
  /// Push a DW_OP_piece / DW_OP_bit_piece for emitting later, if one is needed
  /// to represent a subregister.
  void setSubRegisterPiece(unsigned SizeInBits, unsigned OffsetInBits) {
    assert(SizeInBits < 65536 && OffsetInBits < 65536);
    SubRegisterSizeInBits = SizeInBits;
    SubRegisterOffsetInBits = OffsetInBits;
  }

  /// Add masking operations to stencil out a subregister.
  void maskSubRegister();

  /// Output a dwarf operand and an optional assembler comment.
  virtual void emitOp(uint8_t Op, const char *Comment = nullptr) = 0;

  /// Emit a raw signed value.
  virtual void emitSigned(int64_t Value) = 0;

  /// Emit a raw unsigned value.
  virtual void emitUnsigned(uint64_t Value) = 0;

  virtual void emitData1(uint8_t Value) = 0;

  virtual void emitBaseTypeRef(uint64_t Idx) = 0;

  /// Start emitting data to the temporary buffer. The data stored in the
  /// temporary buffer can be committed to the main output using
  /// commitTemporaryBuffer().
  virtual void enableTemporaryBuffer() = 0;

  /// Disable emission to the temporary buffer. This does not commit data
  /// in the temporary buffer to the main output.
  virtual void disableTemporaryBuffer() = 0;

  /// Return the emitted size, in number of bytes, for the data stored in the
  /// temporary buffer.
  virtual unsigned getTemporaryBufferSize() = 0;

  /// Commit the data stored in the temporary buffer to the main output.
  virtual void commitTemporaryBuffer() = 0;

  /// Emit a normalized unsigned constant.
  void emitConstu(uint64_t Value);

  /// Return whether the given machine register is the frame register in the
  /// current function.
  virtual bool isFrameRegister(const TargetRegisterInfo &TRI, unsigned MachineReg) = 0;

  /// Emit a DW_OP_reg operation. Note that this is only legal inside a DWARF
  /// register location description.
  void addReg(int DwarfReg, const char *Comment = nullptr);

  /// Emit a DW_OP_breg operation.
  void addBReg(int DwarfReg, int Offset);

  /// Emit DW_OP_fbreg <Offset>.
  void addFBReg(int Offset);

  /// Emit a partial DWARF register operation.
  ///
  /// \param MachineReg           The register number.
  /// \param MaxSize              If the register must be composed from
  ///                             sub-registers this is an upper bound
  ///                             for how many bits the emitted DW_OP_piece
  ///                             may cover.
  ///
  /// If size and offset is zero an operation for the entire register is
  /// emitted: Some targets do not provide a DWARF register number for every
  /// register.  If this is the case, this function will attempt to emit a DWARF
  /// register by emitting a fragment of a super-register or by piecing together
  /// multiple subregisters that alias the register.
  ///
  /// \return false if no DWARF register exists for MachineReg.
  bool addMachineReg(const TargetRegisterInfo &TRI, unsigned MachineReg,
                     unsigned MaxSize = ~1U);

  /// Emit a DW_OP_piece or DW_OP_bit_piece operation for a variable fragment.
  /// \param OffsetInBits    This is an optional offset into the location that
  /// is at the top of the DWARF stack.
  void addOpPiece(unsigned SizeInBits, unsigned OffsetInBits = 0);

  /// Emit a shift-right dwarf operation.
  void addShr(unsigned ShiftBy);

  /// Emit a bitwise and dwarf operation.
  void addAnd(unsigned Mask);

  /// Emit a DW_OP_stack_value, if supported.
  ///
  /// The proper way to describe a constant value is DW_OP_constu <const>,
  /// DW_OP_stack_value.  Unfortunately, DW_OP_stack_value was not available
  /// until DWARF 4, so we will continue to generate DW_OP_constu <const> for
  /// DWARF 2 and DWARF 3. Technically, this is incorrect since DW_OP_const
  /// <const> actually describes a value at a constant address, not a constant
  /// value.  However, in the past there was no better way to describe a
  /// constant value, so the producers and consumers started to rely on
  /// heuristics to disambiguate the value vs. location status of the
  /// expression.  See PR21176 for more details.
  void addStackValue();

  /// Finalize an entry value by emitting its size operand, and committing the
  /// DWARF block which has been emitted to the temporary buffer.
  void finalizeEntryValue();

  ~DwarfExpression() = default;

public:
  DwarfExpression(unsigned DwarfVersion, DwarfCompileUnit &CU)
      : CU(CU), SubRegisterSizeInBits(0), SubRegisterOffsetInBits(0),
        LocationKind(Unknown), LocationFlags(Unknown),
        DwarfVersion(DwarfVersion) {}

  /// This needs to be called last to commit any pending changes.
  void finalize();

  /// Emit a signed constant.
  void addSignedConstant(int64_t Value);

  /// Emit an unsigned constant.
  void addUnsignedConstant(uint64_t Value);

  /// Emit an unsigned constant.
  void addUnsignedConstant(const APInt &Value);

  /// Lock this down to become a memory location description.
  void setMemoryLocationKind() {
    assert(isUnknownLocation());
    LocationKind = Memory;
  }

  /// Lock this down to become an entry value location.
  void setEntryValueFlag() {
    LocationFlags |= EntryValue;
  }

  /// Lock this down to become a call site parameter location.
  void setCallSiteParamValueFlag() {
    LocationFlags |= CallSiteParamValue;
  }

  /// Emit a machine register location. As an optimization this may also consume
  /// the prefix of a DwarfExpression if a more efficient representation for
  /// combining the register location and the first operation exists.
  ///
  /// \param FragmentOffsetInBits     If this is one fragment out of a
  /// fragmented
  ///                                 location, this is the offset of the
  ///                                 fragment inside the entire variable.
  /// \return                         false if no DWARF register exists
  ///                                 for MachineReg.
  bool addMachineRegExpression(const TargetRegisterInfo &TRI,
                               DIExpressionCursor &Expr, unsigned MachineReg,
                               unsigned FragmentOffsetInBits = 0);

  /// Begin emission of an entry value dwarf operation. The entry value's
  /// first operand is the size of the DWARF block (its second operand),
  /// which needs to be calculated at time of emission, so we don't emit
  /// any operands here.
  void beginEntryValueExpression(DIExpressionCursor &ExprCursor);

  /// Emit all remaining operations in the DIExpressionCursor.
  ///
  /// \param FragmentOffsetInBits     If this is one fragment out of multiple
  ///                                 locations, this is the offset of the
  ///                                 fragment inside the entire variable.
  void addExpression(DIExpressionCursor &&Expr,
                     unsigned FragmentOffsetInBits = 0);

  /// If applicable, emit an empty DW_OP_piece / DW_OP_bit_piece to advance to
  /// the fragment described by \c Expr.
  void addFragmentOffset(const DIExpression *Expr);

  void emitLegacySExt(unsigned FromBits);
  void emitLegacyZExt(unsigned FromBits);

  /// Emit location information expressed via WebAssembly location + offset
  /// The Index is an identifier for locals, globals or operand stack.
  void addWasmLocation(unsigned Index, int64_t Offset);
};

/// DwarfExpression implementation for .debug_loc entries.
class DebugLocDwarfExpression final : public DwarfExpression {

  struct TempBuffer {
    SmallString<32> Bytes;
    std::vector<std::string> Comments;
    BufferByteStreamer BS;

    TempBuffer(bool GenerateComments) : BS(Bytes, Comments, GenerateComments) {}
  };

  std::unique_ptr<TempBuffer> TmpBuf;
  BufferByteStreamer &OutBS;
  bool IsBuffering = false;

  /// Return the byte streamer that currently is being emitted to.
  ByteStreamer &getActiveStreamer() { return IsBuffering ? TmpBuf->BS : OutBS; }

  void emitOp(uint8_t Op, const char *Comment = nullptr) override;
  void emitSigned(int64_t Value) override;
  void emitUnsigned(uint64_t Value) override;
  void emitData1(uint8_t Value) override;
  void emitBaseTypeRef(uint64_t Idx) override;

  void enableTemporaryBuffer() override;
  void disableTemporaryBuffer() override;
  unsigned getTemporaryBufferSize() override;
  void commitTemporaryBuffer() override;

  bool isFrameRegister(const TargetRegisterInfo &TRI,
                       unsigned MachineReg) override;
public:
  DebugLocDwarfExpression(unsigned DwarfVersion, BufferByteStreamer &BS,
                          DwarfCompileUnit &CU)
      : DwarfExpression(DwarfVersion, CU), OutBS(BS) {}
};

/// DwarfExpression implementation for singular DW_AT_location.
class DIEDwarfExpression final : public DwarfExpression {
  const AsmPrinter &AP;
  DIELoc &OutDIE;
  DIELoc TmpDIE;
  bool IsBuffering = false;

  /// Return the DIE that currently is being emitted to.
  DIELoc &getActiveDIE() { return IsBuffering ? TmpDIE : OutDIE; }

  void emitOp(uint8_t Op, const char *Comment = nullptr) override;
  void emitSigned(int64_t Value) override;
  void emitUnsigned(uint64_t Value) override;
  void emitData1(uint8_t Value) override;
  void emitBaseTypeRef(uint64_t Idx) override;

  void enableTemporaryBuffer() override;
  void disableTemporaryBuffer() override;
  unsigned getTemporaryBufferSize() override;
  void commitTemporaryBuffer() override;

  bool isFrameRegister(const TargetRegisterInfo &TRI,
                       unsigned MachineReg) override;
public:
  DIEDwarfExpression(const AsmPrinter &AP, DwarfCompileUnit &CU, DIELoc &DIE);

  DIELoc *finalize() {
    DwarfExpression::finalize();
    return &OutDIE;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
