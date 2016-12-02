//==-- WebAssemblyTargetStreamer.h - WebAssembly Target Streamer -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares WebAssembly-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYTARGETSTREAMER_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_MCTARGETDESC_WEBASSEMBLYTARGETSTREAMER_H

#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class MCELFStreamer;

/// WebAssembly-specific streamer interface, to implement support
/// WebAssembly-specific assembly directives.
class WebAssemblyTargetStreamer : public MCTargetStreamer {
public:
  explicit WebAssemblyTargetStreamer(MCStreamer &S);

  /// .param
  virtual void emitParam(ArrayRef<MVT> Types) = 0;
  /// .result
  virtual void emitResult(ArrayRef<MVT> Types) = 0;
  /// .local
  virtual void emitLocal(ArrayRef<MVT> Types) = 0;
  /// .endfunc
  virtual void emitEndFunc() = 0;
  /// .functype
  virtual void emitIndirectFunctionType(StringRef name,
                                        SmallVectorImpl<MVT> &Params,
                                        SmallVectorImpl<MVT> &Results) {
    llvm_unreachable("emitIndirectFunctionType not implemented");
  }
  /// .indidx
  virtual void emitIndIdx(const MCExpr *Value) = 0;
  /// .import_global
  virtual void emitGlobalImport(StringRef name) = 0;
};

/// This part is for ascii assembly output
class WebAssemblyTargetAsmStreamer final : public WebAssemblyTargetStreamer {
  formatted_raw_ostream &OS;

public:
  WebAssemblyTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void emitParam(ArrayRef<MVT> Types) override;
  void emitResult(ArrayRef<MVT> Types) override;
  void emitLocal(ArrayRef<MVT> Types) override;
  void emitEndFunc() override;
  void emitIndirectFunctionType(StringRef name,
                                SmallVectorImpl<MVT> &Params,
                                SmallVectorImpl<MVT> &Results) override;
  void emitIndIdx(const MCExpr *Value) override;
  void emitGlobalImport(StringRef name) override;
};

/// This part is for ELF object output
class WebAssemblyTargetELFStreamer final : public WebAssemblyTargetStreamer {
public:
  explicit WebAssemblyTargetELFStreamer(MCStreamer &S);

  void emitParam(ArrayRef<MVT> Types) override;
  void emitResult(ArrayRef<MVT> Types) override;
  void emitLocal(ArrayRef<MVT> Types) override;
  void emitEndFunc() override;
  void emitIndirectFunctionType(StringRef name,
                                SmallVectorImpl<MVT> &Params,
                                SmallVectorImpl<MVT> &Results) override;
  void emitIndIdx(const MCExpr *Value) override;
  void emitGlobalImport(StringRef name) override;
};

} // end namespace llvm

#endif
