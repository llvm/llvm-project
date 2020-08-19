//===-- RegisterInfoPOSIX_arm64.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIX_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_REGISTERINFOPOSIX_ARM64_H

#include "RegisterInfoAndSetInterface.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/lldb-private.h"
#include <map>

enum class SVEState { Unknown, Disabled, FPSIMD, Full };

class RegisterInfoPOSIX_arm64
    : public lldb_private::RegisterInfoAndSetInterface {
public:
  enum { GPRegSet = 0, FPRegSet, SVERegSet };

  // AArch64 Register set FP/SIMD feature configuration
  enum {
    eVectorQuadwordAArch64,
    eVectorQuadwordAArch64SVE,
    eVectorQuadwordAArch64SVEMax = 256
  };

  // based on RegisterContextDarwin_arm64.h
  struct GPR {
    uint64_t x[29]; // x0-x28
    uint64_t fp;    // x29
    uint64_t lr;    // x30
    uint64_t sp;    // x31
    uint64_t pc;    // pc
    uint32_t cpsr;  // cpsr
  };

  // based on RegisterContextDarwin_arm64.h
  struct VReg {
    uint8_t bytes[16];
  };

  // based on RegisterContextDarwin_arm64.h
  struct FPU {
    VReg v[32];
    uint32_t fpsr;
    uint32_t fpcr;
  };

  // based on RegisterContextDarwin_arm64.h
  struct EXC {
    uint64_t far;       // Virtual Fault Address
    uint32_t esr;       // Exception syndrome
    uint32_t exception; // number of arm exception token
  };

  // based on RegisterContextDarwin_arm64.h
  struct DBG {
    uint64_t bvr[16];
    uint64_t bcr[16];
    uint64_t wvr[16];
    uint64_t wcr[16];
    uint64_t mdscr_el1;
  };

  RegisterInfoPOSIX_arm64(const lldb_private::ArchSpec &target_arch);

  size_t GetGPRSize() const override;

  size_t GetFPRSize() const override;

  const lldb_private::RegisterInfo *GetRegisterInfo() const override;

  uint32_t GetRegisterCount() const override;

  const lldb_private::RegisterSet *
  GetRegisterSet(size_t reg_set) const override;

  size_t GetRegisterSetCount() const override;

  size_t GetRegisterSetFromRegisterIndex(uint32_t reg_index) const override;

  uint32_t ConfigureVectorRegisterInfos(uint32_t mode);

  bool VectorSizeIsValid(uint32_t vq) {
    if (vq >= eVectorQuadwordAArch64 && vq <= eVectorQuadwordAArch64SVEMax)
      return true;
    return false;
  }

  bool IsSVEEnabled() const { return m_vector_reg_vq > eVectorQuadwordAArch64; }

  bool IsSVEZReg(unsigned reg) const;
  bool IsSVEPReg(unsigned reg) const;
  bool IsSVERegVG(unsigned reg) const;

  uint32_t GetRegNumSVEZ0() const;
  uint32_t GetRegNumSVEFFR() const;
  uint32_t GetRegNumFPCR() const;
  uint32_t GetRegNumFPSR() const;

private:
  typedef std::map<uint32_t, std::vector<lldb_private::RegisterInfo>>
      per_vq_register_infos;

  per_vq_register_infos m_per_vq_reg_infos;

  uint32_t m_vector_reg_vq = eVectorQuadwordAArch64;

  const lldb_private::RegisterInfo *m_register_info_p;
  uint32_t m_register_info_count;
};

#endif
