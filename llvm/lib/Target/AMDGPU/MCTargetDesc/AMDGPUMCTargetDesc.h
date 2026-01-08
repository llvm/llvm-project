//===-- AMDGPUMCTargetDesc.h - AMDGPU Target Descriptions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Provides AMDGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCTARGETDESC_H

#include "llvm/MC/MCInstrAnalysis.h"
#include <cstdint>
#include <memory>

namespace llvm {
class Target;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;

enum AMDGPUDwarfFlavour : unsigned { Wave64 = 0, Wave32 = 1 };

MCRegisterInfo *createGCNMCRegisterInfo(AMDGPUDwarfFlavour DwarfFlavour);

MCCodeEmitter *createAMDGPUMCCodeEmitter(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createAMDGPUAsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter>
createAMDGPUELFObjectWriter(bool Is64Bit, uint8_t OSABI,
                            bool HasRelocationAddend);

namespace AMDGPU {
class AMDGPUMCInstrAnalysis : public MCInstrAnalysis {
private:
  unsigned VgprMSBs = 0;

public:
  explicit AMDGPUMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override;

  void resetState() override { VgprMSBs = 0; }

  void updateState(const MCInst &Inst, uint64_t Addr) override;

  unsigned getVgprMSBs() const { return VgprMSBs; }
};

} // namespace AMDGPU

} // namespace llvm

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "AMDGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"

#endif
