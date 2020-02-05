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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/Support/Endian.h"
#include <cstdint>

namespace llvm {

class MCAsmLayout;
class MCAssembler;
class MCCFIInstruction;
struct MCFixupKindInfo;
class MCFragment;
class MCInst;
class MCObjectStreamer;
class MCObjectTargetWriter;
class MCObjectWriter;
class MCRelaxableFragment;
class MCSubtargetInfo;
class MCValue;
class raw_pwrite_stream;

/// Generic interface to target specific assembler backends.
class MCAsmBackend {
protected: // Can only create subclasses.
  MCAsmBackend(support::endianness Endian);

public:
  MCAsmBackend(const MCAsmBackend &) = delete;
  MCAsmBackend &operator=(const MCAsmBackend &) = delete;
  virtual ~MCAsmBackend();

  const support::endianness Endian;

  /// Return true if this target might automatically pad instructions and thus
  /// need to emit padding enable/disable directives around sensative code.
  virtual bool allowAutoPadding() const { return false; }

  /// Give the target a chance to manipulate state related to instruction
  /// alignment (e.g. padding for optimization) before and after actually
  /// emitting the instruction.
  virtual void alignBranchesBegin(MCObjectStreamer &OS, const MCInst &Inst) {}
  virtual void alignBranchesEnd(MCObjectStreamer &OS, const MCInst &Inst) {}

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

  /// Get the number of target specific fixup kinds.
  virtual unsigned getNumFixupKinds() const = 0;

  /// Map a relocation name used in .reloc to a fixup kind.
  virtual Optional<MCFixupKind> getFixupKind(StringRef Name) const;

  /// Get information on a fixup kind.
  virtual const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const;

  /// Hook to check if a relocation is needed for some target specific reason.
  virtual bool shouldForceRelocation(const MCAssembler &Asm,
                                     const MCFixup &Fixup,
                                     const MCValue &Target) {
    return false;
  }

  /// Hook to check if extra nop bytes must be inserted for alignment directive.
  /// For some targets this may be necessary in order to support linker
  /// relaxation. The number of bytes to insert are returned in Size.
  virtual bool shouldInsertExtraNopBytesForCodeAlign(const MCAlignFragment &AF,
                                                     unsigned &Size) {
    return false;
  }

  /// Hook which indicates if the target requires a fixup to be generated when
  /// handling an align directive in an executable section
  virtual bool shouldInsertFixupForCodeAlign(MCAssembler &Asm,
                                             const MCAsmLayout &Layout,
                                             MCAlignFragment &AF) {
    return false;
  }

  virtual bool evaluateTargetFixup(const MCAssembler &Asm,
                                   const MCAsmLayout &Layout,
                                   const MCFixup &Fixup, const MCFragment *DF,
                                   const MCValue &Target, uint64_t &Value,
                                   bool &WasForced) {
    llvm_unreachable("Need to implement hook if target has custom fixups");
  }

  /// Apply the \p Value for given \p Fixup into the provided data fragment, at
  /// the offset specified by the fixup and following the fixup kind as
  /// appropriate. Errors (such as an out of range fixup value) should be
  /// reported via \p Ctx.
  /// The  \p STI is present only for fragments of type MCRelaxableFragment and
  /// MCDataFragment with hasInstructions() == true.
  virtual void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                          const MCValue &Target, MutableArrayRef<char> Data,
                          uint64_t Value, bool IsResolved,
                          const MCSubtargetInfo *STI) const = 0;

  /// Check whether the given target requires emitting differences of two
  /// symbols as a set of relocations.
  virtual bool requiresDiffExpressionRelocations() const { return false; }

  /// @}

  /// \name Target Relaxation Interfaces
  /// @{

  /// Check whether the given instruction may need relaxation.
  ///
  /// \param Inst - The instruction to test.
  /// \param STI - The MCSubtargetInfo in effect when the instruction was
  /// encoded.
  virtual bool mayNeedRelaxation(const MCInst &Inst,
                                 const MCSubtargetInfo &STI) const = 0;

  /// Target specific predicate for whether a given fixup requires the
  /// associated instruction to be relaxed.
  virtual bool fixupNeedsRelaxationAdvanced(const MCFixup &Fixup, bool Resolved,
                                            uint64_t Value,
                                            const MCRelaxableFragment *DF,
                                            const MCAsmLayout &Layout,
                                            const bool WasForced) const;

  /// Simple predicate for targets where !Resolved implies requiring relaxation
  virtual bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout) const = 0;

  /// Relax the instruction in the given fragment to the next wider instruction.
  ///
  /// \param Inst The instruction to relax, which may be the same as the
  /// output.
  /// \param STI the subtarget information for the associated instruction.
  /// \param [out] Res On return, the relaxed instruction.
  virtual void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                                MCInst &Res) const = 0;

  /// @}

  /// Returns the minimum size of a nop in bytes on this target. The assembler
  /// will use this to emit excess padding in situations where the padding
  /// required for simple alignment would be less than the minimum nop size.
  ///
  virtual unsigned getMinimumNopSize() const { return 1; }

  /// Write an (optimal) nop sequence of Count bytes to the given output. If the
  /// target cannot generate such a sequence, it should return an error.
  ///
  /// \return - True on success.
  virtual bool writeNopData(raw_ostream &OS, uint64_t Count) const = 0;

  /// Give backend an opportunity to finish layout after relaxation
  virtual void finishLayout(MCAssembler const &Asm,
                            MCAsmLayout &Layout) const {}

  /// Handle any target-specific assembler flags. By default, do nothing.
  virtual void handleAssemblerFlag(MCAssemblerFlag Flag) {}

  /// Generate the compact unwind encoding for the CFI instructions.
  virtual uint32_t
      generateCompactUnwindEncoding(ArrayRef<MCCFIInstruction>) const {
    return 0;
  }

  /// Check whether a given symbol has been flagged with MICROMIPS flag.
  virtual bool isMicroMips(const MCSymbol *Sym) const {
    return false;
  }
};

} // end namespace llvm

#endif // LLVM_MC_MCASMBACKEND_H
