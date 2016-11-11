//===-- AMDGPUTargetStreamer.h - AMDGPU Target Streamer --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUTARGETSTREAMER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUTARGETSTREAMER_H

#include "AMDKernelCodeT.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
#include "AMDGPURuntimeMetadata.h"

class DataLayout;
class Function;
class MCELFStreamer;
class MCSymbol;
class MDNode;
class Module;
class Type;

class AMDGPUTargetStreamer : public MCTargetStreamer {
public:
  AMDGPUTargetStreamer(MCStreamer &S);
  virtual void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                 uint32_t Minor) = 0;

  virtual void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
                                             uint32_t Stepping,
                                             StringRef VendorName,
                                             StringRef ArchName) = 0;

  virtual void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) = 0;

  virtual void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) = 0;

  virtual void EmitAMDGPUHsaModuleScopeGlobal(StringRef GlobalName) = 0;

  virtual void EmitAMDGPUHsaProgramScopeGlobal(StringRef GlobalName) = 0;

  /// Emit runtime metadata as a note element.
  void emitRuntimeMetadataAsNoteElement(Module &M);

private:
  void emitRuntimeMetadata(Module &M);
  void emitStartOfRuntimeMetadata(const Module &M);

  /// Emit runtime metadata for a kernel function.
  void emitRuntimeMetadata(const Function &F);

  // Emit runtime metadata for a kernel argument.
  void emitRuntimeMetadataForKernelArg(const DataLayout &DL,
      Type *T, AMDGPU::RuntimeMD::KernelArg::Kind Kind,
      StringRef BaseTypeName = "", StringRef TypeName = "",
      StringRef ArgName = "", StringRef TypeQual = "",
      StringRef AccQual = "");

  /// Emit a key and an integer value for runtime metadata.
  void emitRuntimeMDIntValue(AMDGPU::RuntimeMD::Key K,
      uint64_t V, unsigned Size);

  /// Emit a key and a string value for runtime metadata.
  void emitRuntimeMDStringValue(AMDGPU::RuntimeMD::Key K,
      StringRef S);

  /// Emit a key and three integer values for runtime metadata.
  /// The three integer values are obtained from MDNode \p Node;
  void emitRuntimeMDThreeIntValues(AMDGPU::RuntimeMD::Key K, MDNode *Node,
                                   unsigned Size);
};

class AMDGPUTargetAsmStreamer : public AMDGPUTargetStreamer {
  formatted_raw_ostream &OS;
public:
  AMDGPUTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
                                     uint32_t Stepping, StringRef VendorName,
                                     StringRef ArchName) override;

  void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void EmitAMDGPUHsaModuleScopeGlobal(StringRef GlobalName) override;

  void EmitAMDGPUHsaProgramScopeGlobal(StringRef GlobalName) override;
};

class AMDGPUTargetELFStreamer : public AMDGPUTargetStreamer {
  MCStreamer &Streamer;

public:
  AMDGPUTargetELFStreamer(MCStreamer &S);

  MCELFStreamer &getStreamer();

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
                                     uint32_t Stepping, StringRef VendorName,
                                     StringRef ArchName) override;

  void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void EmitAMDGPUHsaModuleScopeGlobal(StringRef GlobalName) override;

  void EmitAMDGPUHsaProgramScopeGlobal(StringRef GlobalName) override;
};

}
#endif
