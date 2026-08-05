//=====-- NVPTXTargetStreamer.h - NVPTX Target Streamer ------*- C++ -*--=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXTARGETSTREAMER_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MCSection;
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
};

} // end namespace llvm

#endif
