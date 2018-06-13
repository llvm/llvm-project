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
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"

namespace llvm {
#include "AMDGPUPTNote.h"

class DataLayout;
class Function;
class MCELFStreamer;
class MCSymbol;
class MDNode;
class Module;
class Type;

class AMDGPUTargetStreamer : public MCTargetStreamer {
protected:
  MCContext &getContext() const { return Streamer.getContext(); }

  /// \returns Equivalent EF_AMDGPU_MACH_* value for given \p GPU name.
  unsigned getMACH(StringRef GPU) const;

public:
  /// \returns Equivalent GPU name for an EF_AMDGPU_MACH_* value.
  static const char *getMachName(unsigned Mach);

  AMDGPUTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

  virtual void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                 uint32_t Minor) = 0;

  virtual void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
                                             uint32_t Stepping,
                                             StringRef VendorName,
                                             StringRef ArchName) = 0;

  virtual void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) = 0;

  virtual void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) = 0;

  /// \returns True on success, false on failure.
  virtual bool EmitISAVersion(StringRef IsaVersionString) = 0;

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(StringRef HSAMetadataString);

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) = 0;

  /// \returns True on success, false on failure.
  virtual bool EmitPALMetadata(const AMDGPU::PALMD::Metadata &PALMetadata) = 0;

  virtual void EmitAmdhsaKernelDescriptor(
      StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor) = 0;
};

class AMDGPUTargetAsmStreamer final : public AMDGPUTargetStreamer {
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

  /// \returns True on success, false on failure.
  bool EmitISAVersion(StringRef IsaVersionString) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitPALMetadata(const AMDGPU::PALMD::Metadata &PALMetadata) override;

  void EmitAmdhsaKernelDescriptor(
      StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor) override;
};

class AMDGPUTargetELFStreamer final : public AMDGPUTargetStreamer {
  MCStreamer &Streamer;

  void EmitAMDGPUNote(const MCExpr *DescSize, unsigned NoteType,
                      function_ref<void(MCELFStreamer &)> EmitDesc);

public:
  AMDGPUTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  MCELFStreamer &getStreamer();

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
                                     uint32_t Stepping, StringRef VendorName,
                                     StringRef ArchName) override;

  void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  /// \returns True on success, false on failure.
  bool EmitISAVersion(StringRef IsaVersionString) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitPALMetadata(const AMDGPU::PALMD::Metadata &PALMetadata) override;

  void EmitAmdhsaKernelDescriptor(
      StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor) override;
};

}
#endif
