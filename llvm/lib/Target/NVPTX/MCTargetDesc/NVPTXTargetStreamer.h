//=====-- NVPTXTargetStreamer.h - NVPTX Target Streamer ------*- C++ -*--=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXTARGETSTREAMER_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXTARGETSTREAMER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCStreamer.h"
#include <optional>

namespace llvm {
class MCSection;
class MCSymbol;
class formatted_raw_ostream;

/// Implments NVPTX-specific streamer.
class NVPTXTargetStreamer : public MCTargetStreamer {
private:
  SmallVector<std::string, 4> DwarfFiles;
  bool HasSections = false;

public:
  NVPTXTargetStreamer(MCStreamer &S);
  ~NVPTXTargetStreamer() override;

  /// Emit the banner which specifies details of PTX generator.
  virtual void emitBanner() {}

  /// Emit the PTX ISA version number.
  virtual void emitVersionDirective(unsigned PTXVersion) {}

  /// Emit architecture and platform target.
  virtual void emitTargetDirective(StringRef Target, bool TexModeIndependent,
                                   bool HasDebug) {}

  /// Emit address size used for this PTX module.
  virtual void emitAddressSizeDirective(unsigned AddrSize) {}

  /// Emit the list of branch targets a brx.idx may jump to.
  virtual void emitBranchTargetsDirective(ArrayRef<const MCSymbol *> Targets) {}

  /// Declare a register \p SizeInBits wide. \p Count declares a numbered bank
  /// of that many registers sharing \p Name as their prefix, rather than a
  /// single register named \p Name.
  virtual void emitRegDirective(unsigned SizeInBits, StringRef Name,
                                std::optional<unsigned> Count = std::nullopt) {}

  /// Declare \p Size bytes of local (stack) memory named \p Name.
  virtual void emitLocalDirective(Align Alignment, const MCSymbol *Name,
                                  uint64_t Size) {}

  /// Emit an alias from \p Name to \p Aliasee.
  virtual void emitAliasDirective(const MCSymbol *Name,
                                  const MCSymbol *Aliasee) {}

  /// Emit a pragma governing the code that follows it.
  virtual void emitPragmaDirective(StringRef Pragma) {}

  /// Emit a section with an empty body.
  virtual void emitEmptySectionDirective(StringRef Name) {}

  /// Outputs the list of the DWARF '.file' directives to the streamer.
  void outputDwarfFileDirectives();
  /// Close last section.
  void closeLastSection();

  /// Record DWARF file directives for later output.
  /// According to PTX ISA, CUDA Toolkit documentation, 11.5.3. Debugging
  /// Directives: .file
  /// (http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#debugging-directives-file),
  /// The .file directive is allowed only in the outermost scope, i.e., at the
  /// same level as kernel and device function declarations. Also, the order of
  /// the .loc and .file directive does not matter, .file directives may follow
  /// the .loc directives where the file is referenced.
  /// LLVM emits .file directives immediately the location debug info is
  /// emitted, i.e. they may be emitted inside functions. We gather all these
  /// directives and emit them outside of the sections and, thus, outside of the
  /// functions.
  void emitDwarfFileDirective(StringRef Directive) override;
  void changeSection(const MCSection *CurSection, MCSection *Section,
                     uint32_t SubSection, raw_ostream &OS) override;
  /// Emit the bytes in \p Data into the output.
  ///
  /// This is used to emit bytes in \p Data as sequence of .byte directives.
  void emitRawBytes(StringRef Data) override;
  /// Makes sure that labels are mangled the same way as the actual symbols.
  void emitValue(const MCExpr *Value) override;
};

class NVPTXAsmTargetStreamer : public NVPTXTargetStreamer {
  formatted_raw_ostream &OS;

public:
  NVPTXAsmTargetStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  ~NVPTXAsmTargetStreamer() override;

  void emitBanner() override;

  void emitVersionDirective(unsigned PTXVersion) override;

  void emitTargetDirective(StringRef Target, bool TexModeIndependent,
                           bool HasDebug) override;

  void emitAddressSizeDirective(unsigned AddrSize) override;

  void emitBranchTargetsDirective(ArrayRef<const MCSymbol *> Targets) override;

  void emitRegDirective(unsigned SizeInBits, StringRef Name,
                        std::optional<unsigned> Count) override;

  void emitLocalDirective(Align Alignment, const MCSymbol *Name,
                          uint64_t Size) override;

  void emitAliasDirective(const MCSymbol *Name,
                          const MCSymbol *Aliasee) override;

  void emitPragmaDirective(StringRef Pragma) override;

  void emitEmptySectionDirective(StringRef Name) override;
};

} // end namespace llvm

#endif
