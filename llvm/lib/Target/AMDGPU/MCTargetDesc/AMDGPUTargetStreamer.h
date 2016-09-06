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

class MCELFStreamer;
class MCSymbol;

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

  enum NoteType {
    NT_AMDGPU_HSA_CODE_OBJECT_VERSION = 1,
    NT_AMDGPU_HSA_HSAIL = 2,
    NT_AMDGPU_HSA_ISA = 3,
    NT_AMDGPU_HSA_PRODUCER = 4,
    NT_AMDGPU_HSA_PRODUCER_OPTIONS = 5,
    NT_AMDGPU_HSA_EXTENSION = 6,
    NT_AMDGPU_HSA_HLDEBUG_DEBUG = 101,
    NT_AMDGPU_HSA_HLDEBUG_TARGET = 102
  };

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
