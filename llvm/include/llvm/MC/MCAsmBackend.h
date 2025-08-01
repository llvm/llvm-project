//===- llvm/MC/MCAsmBackend.h - MC Asm Backend ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMBACKEND_H
#define LLVM_MC_MCASMBACKEND_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include <cstdint>

namespace llvm {

class MCFragment;
class MCSymbol;
class MCAssembler;
class MCContext;
struct MCDwarfFrameInfo;
class MCInst;
class MCObjectStreamer;
class MCObjectTargetWriter;
class MCObjectWriter;
class MCOperand;
class MCSubtargetInfo;
class MCValue;
class raw_pwrite_stream;
class StringRef;
class raw_ostream;

/// Target independent information on a fixup kind.
struct MCFixupKindInfo {
  /// A target specific name for the fixup kind. The names will be unique for
  /// distinct kinds on any given target.
  const char *Name;

  /// The bit offset to write the relocation into.
  uint8_t TargetOffset;

  /// The number of bits written by this fixup. The bits are assumed to be
  /// contiguous.
  uint8_t TargetSize;

  /// Flags describing additional information on this fixup kind.
  unsigned Flags;
};

/// Generic interface to target specific assembler backends.
class LLVM_ABI MCAsmBackend {
protected: // Can only create subclasses.
  MCAsmBackend(llvm::endianness Endian) : Endian(Endian) {}

  MCAssembler *Asm = nullptr;

  bool AllowAutoPadding = false;
  bool AllowEnhancedRelaxation = false;

public:
  MCAsmBackend(const MCAsmBackend &) = delete;
  MCAsmBackend &operator=(const MCAsmBackend &) = delete;
  virtual ~MCAsmBackend();

  const llvm::endianness Endian;

  void setAssembler(MCAssembler *A) { Asm = A; }

  MCContext &getContext() const;

  /// Return true if this target might automatically pad instructions and thus
  /// need to emit padding enable/disable directives around sensative code.
  bool allowAutoPadding() const { return AllowAutoPadding; }
  /// Return true if this target allows an unrelaxable instruction to be
  /// emitted into RelaxableFragment and then we can increase its size in a
  /// tricky way for optimization.
  bool allowEnhancedRelaxation() const { return AllowEnhancedRelaxation; }

  /// lifetime management
  virtual void reset() {}

  /// Create a new MCObjectWriter instance for use by the assembler backend to
  /// emit the final object file.
  std::unique_ptr<MCObjectWriter>
  createObjectWriter(raw_pwrite_stream &OS) const;

  /// Create an MCObjectWriter that writes two object files: a .o file which is
  /// linked into the final program and a .dwo file which is used by debuggers.
  /// This function is only supported with ELF targets.
  std::unique_ptr<MCObjectWriter>
  createDwoObjectWriter(raw_pwrite_stream &OS, raw_pwrite_stream &DwoOS) const;

  virtual std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const = 0;

  /// \name Target Fixup Interfaces
  /// @{

  /// Map a relocation name used in .reloc to a fixup kind.
  virtual std::optional<MCFixupKind> getFixupKind(StringRef Name) const;

  /// Get information on a fixup kind.
  virtual MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const;

  // Evaluate a fixup, returning std::nullopt to use default handling for
  // `Value` and `IsResolved`. Otherwise, returns `IsResolved` with the
  // expectation that the hook updates `Value`.
  virtual std::optional<bool> evaluateFixup(const MCFragment &, MCFixup &,
                                            MCValue &, uint64_t &) {
    return {};
  }

  void maybeAddReloc(const MCFragment &, const MCFixup &, const MCValue &,
                     uint64_t &Value, bool IsResolved);

  /// Determine if a relocation is required. In addition,
  /// Apply the \p Value for given \p Fixup into the provided data fragment, at
  /// the offset specified by the fixup and following the fixup kind as
  /// appropriate. Errors (such as an out of range fixup value) should be
  /// reported via \p Ctx.
  virtual void applyFixup(const MCFragment &, const MCFixup &,
                          const MCValue &Target, MutableArrayRef<char> Data,
                          uint64_t Value, bool IsResolved) = 0;

  /// @}

  /// \name Target Relaxation Interfaces
  /// @{

  /// Check whether the given instruction (encoded as Opcode+Operands) may need
  /// relaxation.
  virtual bool mayNeedRelaxation(unsigned Opcode, ArrayRef<MCOperand> Operands,
                                 const MCSubtargetInfo &STI) const {
    return false;
  }

  /// Target specific predicate for whether a given fixup requires the
  /// associated instruction to be relaxed.
  virtual bool fixupNeedsRelaxationAdvanced(const MCFragment &, const MCFixup &,
                                            const MCValue &, uint64_t,
                                            bool Resolved) const;

  /// Simple predicate for targets where !Resolved implies requiring relaxation
  virtual bool fixupNeedsRelaxation(const MCFixup &Fixup,
                                    uint64_t Value) const {
    llvm_unreachable("Needed if mayNeedRelaxation may return true");
  }

  /// Relax the instruction in the given fragment to the next wider instruction.
  ///
  /// \param [out] Inst The instruction to relax, which is also the relaxed
  /// instruction.
  /// \param STI the subtarget information for the associated instruction.
  virtual void relaxInstruction(MCInst &Inst,
                                const MCSubtargetInfo &STI) const {
    llvm_unreachable(
        "Needed if fixupNeedsRelaxation/fixupNeedsRelaxationAdvanced may "
        "return true");
  }

  // Defined by linker relaxation targets.

  // Return false to use default handling. Otherwise, set `Size` to the number
  // of padding bytes.
  virtual bool relaxAlign(MCFragment &F, unsigned &Size) { return false; }
  virtual bool relaxDwarfLineAddr(MCFragment &, bool &WasRelaxed) const {
    return false;
  }
  virtual bool relaxDwarfCFA(MCFragment &, bool &WasRelaxed) const {
    return false;
  }

  // Defined by linker relaxation targets to possibly emit LEB128 relocations
  // and set Value at the relocated location.
  virtual std::pair<bool, bool> relaxLEB128(MCFragment &,
                                            int64_t &Value) const {
    return std::make_pair(false, false);
  }

  /// @}

  /// Returns the minimum size of a nop in bytes on this target. The assembler
  /// will use this to emit excess padding in situations where the padding
  /// required for simple alignment would be less than the minimum nop size.
  ///
  virtual unsigned getMinimumNopSize() const { return 1; }

  /// Returns the maximum size of a nop in bytes on this target.
  ///
  virtual unsigned getMaximumNopSize(const MCSubtargetInfo &STI) const {
    return 0;
  }

  /// Write an (optimal) nop sequence of Count bytes to the given output. If the
  /// target cannot generate such a sequence, it should return an error.
  ///
  /// \return - True on success.
  virtual bool writeNopData(raw_ostream &OS, uint64_t Count,
                            const MCSubtargetInfo *STI) const = 0;

  // Return true if fragment offsets have been adjusted and an extra layout
  // iteration is needed.
  virtual bool finishLayout(const MCAssembler &Asm) const { return false; }

  /// Generate the compact unwind encoding for the CFI instructions.
  virtual uint64_t generateCompactUnwindEncoding(const MCDwarfFrameInfo *FI,
                                                 const MCContext *Ctxt) const {
    return 0;
  }

  bool isDarwinCanonicalPersonality(const MCSymbol *Sym) const;

  // Return STI for fragments with hasInstructions() == true.
  static const MCSubtargetInfo *getSubtargetInfo(const MCFragment &F);
};

} // end namespace llvm

#endif // LLVM_MC_MCASMBACKEND_H
