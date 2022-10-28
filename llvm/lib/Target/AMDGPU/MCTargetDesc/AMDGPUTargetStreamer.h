//===-- AMDGPUTargetStreamer.h - AMDGPU Target Streamer --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUTARGETSTREAMER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUTARGETSTREAMER_H

#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPUPALMetadata.h"
#include "llvm/MC/MCStreamer.h"

struct amd_kernel_code_t;

namespace llvm {

class MCELFStreamer;
class MCSymbol;
class formatted_raw_ostream;

namespace AMDGPU {
namespace HSAMD {
struct Metadata;
}
} // namespace AMDGPU

namespace amdhsa {
struct kernel_descriptor_t;
}

class AMDGPUTargetStreamer : public MCTargetStreamer {
  AMDGPUPALMetadata PALMetadata;

protected:
  // TODO: Move HSAMetadataStream to AMDGPUTargetStreamer.
  Optional<AMDGPU::IsaInfo::AMDGPUTargetID> TargetID;

  MCContext &getContext() const { return Streamer.getContext(); }

public:
  AMDGPUTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

  AMDGPUPALMetadata *getPALMetadata() { return &PALMetadata; }

  virtual void EmitDirectiveAMDGCNTarget(){};

  virtual void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                 uint32_t Minor){};

  virtual void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                               uint32_t Stepping,
                                               StringRef VendorName,
                                               StringRef ArchName){};

  virtual void EmitAMDKernelCodeT(const amd_kernel_code_t &Header){};

  virtual void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type){};

  virtual void emitAMDGPULDS(MCSymbol *Symbol, unsigned Size, Align Alignment) {
  }

  /// \returns True on success, false on failure.
  virtual bool EmitISAVersion() { return true; }

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadataV2(StringRef HSAMetadataString);

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadataV3(StringRef HSAMetadataString);

  /// Emit HSA Metadata
  ///
  /// When \p Strict is true, known metadata elements must already be
  /// well-typed. When \p Strict is false, known types are inferred and
  /// the \p HSAMetadata structure is updated with the correct types.
  ///
  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) {
    return true;
  }

  /// \returns True on success, false on failure.
  virtual bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) {
    return true;
  }

  /// \returns True on success, false on failure.
  virtual bool EmitCodeEnd(const MCSubtargetInfo &STI) { return true; }

  virtual void EmitAmdhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr){};

  static StringRef getArchNameFromElfMach(unsigned ElfMach);
  static unsigned getElfMach(StringRef GPU);

  const Optional<AMDGPU::IsaInfo::AMDGPUTargetID> &getTargetID() const {
    return TargetID;
  }
  Optional<AMDGPU::IsaInfo::AMDGPUTargetID> &getTargetID() {
    return TargetID;
  }
  void initializeTargetID(const MCSubtargetInfo &STI) {
    assert(TargetID == None && "TargetID can only be initialized once");
    TargetID.emplace(STI);
  }
  void initializeTargetID(const MCSubtargetInfo &STI, StringRef FeatureString) {
    initializeTargetID(STI);

    assert(getTargetID() != None && "TargetID is None");
    getTargetID()->setTargetIDFromFeaturesString(FeatureString);
  }
};

class AMDGPUTargetAsmStreamer final : public AMDGPUTargetStreamer {
  formatted_raw_ostream &OS;
public:
  AMDGPUTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void finish() override;

  void EmitDirectiveAMDGCNTarget() override;

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                       uint32_t Stepping, StringRef VendorName,
                                       StringRef ArchName) override;

  void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void emitAMDGPULDS(MCSymbol *Sym, unsigned Size, Align Alignment) override;

  /// \returns True on success, false on failure.
  bool EmitISAVersion() override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitCodeEnd(const MCSubtargetInfo &STI) override;

  void EmitAmdhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr) override;
};

class AMDGPUTargetELFStreamer final : public AMDGPUTargetStreamer {
  const MCSubtargetInfo &STI;
  MCStreamer &Streamer;

  void EmitNote(StringRef Name, const MCExpr *DescSize, unsigned NoteType,
                function_ref<void(MCELFStreamer &)> EmitDesc);

  unsigned getEFlags();

  unsigned getEFlagsR600();
  unsigned getEFlagsAMDGCN();

  unsigned getEFlagsUnknownOS();
  unsigned getEFlagsAMDHSA();
  unsigned getEFlagsAMDPAL();
  unsigned getEFlagsMesa3D();

  unsigned getEFlagsV3();
  unsigned getEFlagsV4();

public:
  AMDGPUTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  MCELFStreamer &getStreamer();

  void finish() override;

  void EmitDirectiveAMDGCNTarget() override;

  void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                         uint32_t Minor) override;

  void EmitDirectiveHSACodeObjectISAV2(uint32_t Major, uint32_t Minor,
                                       uint32_t Stepping, StringRef VendorName,
                                       StringRef ArchName) override;

  void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void emitAMDGPULDS(MCSymbol *Sym, unsigned Size, Align Alignment) override;

  /// \returns True on success, false on failure.
  bool EmitISAVersion() override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(msgpack::Document &HSAMetadata, bool Strict) override;

  /// \returns True on success, false on failure.
  bool EmitHSAMetadata(const AMDGPU::HSAMD::Metadata &HSAMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitCodeEnd(const MCSubtargetInfo &STI) override;

  void EmitAmdhsaKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const amdhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr) override;
};

}
#endif
