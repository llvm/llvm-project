//===-- llvm/BinaryFormat/SFrame.h ---SFrame Data Structures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains data-structure definitions and constants to support
/// unwinding based on .sframe sections.  This only supports SFRAME_VERSION_2
/// as described at https://sourceware.org/binutils/docs/sframe-spec.html
///
/// Naming conventions follow the spec document. #defines converted to constants
/// and enums for better C++ compatibility.
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_SFRAME_H
#define LLVM_BINARYFORMAT_SFRAME_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

namespace sframe {

constexpr uint16_t SFRAME_MAGIC = 0xDEE2;

enum : uint8_t {
  SFRAME_VERSION_1 = 1,
  SFRAME_VERSION_2 = 2,
};

/// sframe_preable.sfp_flags flags.
enum : uint8_t {
  SFRAME_F_FDE_SORTED = 0x1,
  SFRAME_F_FRAME_POINTER = 0x2,
};

/// Possible values for sframe_header.sfh_abi_arch.
enum : uint8_t {
  SFRAME_ABI_AARCH64_ENDIAN_BIG = 1,
  SFRAME_ABI_AARCH64_ENDIAN_LITTLE = 2,
  SFRAME_ABI_AMD64_ENDIAN_LITTLE = 3
};

/// SFrame FRE Types. Bits 0-3 of sframe_func_desc_entry.sfde_func_info.
enum : uint8_t {
  SFRAME_FRE_TYPE_ADDR1 = 0,
  SFRAME_FRE_TYPE_ADDR2 = 1,
  SFRAME_FRE_TYPE_ADDR4 = 2,
};

/// SFrame FDE Types. Bit 4 of sframe_func_desc_entry.sfde_func_info.
enum : uint8_t {
  SFRAME_FDE_TYPE_PCINC = 0,
  SFRAME_FDE_TYPE_PCMASK = 1,
};

/// Speficies key used for signing return addresses. Bit 5 of
/// sframe_func_desc_entry.sfde_func_info.
enum : uint8_t {
  SFRAME_AARCH64_PAUTH_KEY_A = 0,
  SFRAME_AARCH64_PAUTH_KEY_B = 1,
};

/// Size of stack offsets. Bits 5-6 of sframe_fre_info.fre_info.
enum : uint8_t {
  SFRAME_FRE_OFFSET_1B = 0,
  SFRAME_FRE_OFFSET_2B = 1,
  SFRAME_FRE_OFFSET_4B = 2,
};

/// Stack frame base register. Bit 0 of sframe_fre_info.fre_info.
enum : uint8_t { SFRAME_BASE_REG_FP = 0, SFRAME_BASE_REG_SP = 1 };

LLVM_PACKED_START

struct sframe_preamble {
  uint16_t sfp_magic;
  uint8_t sfp_version;
  uint8_t sfp_flags;
};

struct sframe_header {
  sframe_preamble sfh_preamble;
  uint8_t sfh_abi_arch;
  int8_t sfh_cfa_fixed_fp_offset;
  int8_t sfh_cfa_fixed_ra_offset;
  uint8_t sfh_auxhdr_len;
  uint32_t sfh_num_fdes;
  uint32_t sfh_num_fres;
  uint32_t sfh_fre_len;
  uint32_t sfh_fdeoff;
  uint32_t sfh_freoff;
};

struct sframe_func_desc_entry {
  int32_t sfde_func_start_address;
  uint32_t sfde_func_size;
  uint32_t sfde_func_start_fre_off;
  uint32_t sfde_func_num_fres;
  uint8_t sfde_func_info;
  uint8_t sfde_func_rep_size;
  uint16_t sfde_func_padding2;

  uint8_t getPAuthKey() const { return (sfde_func_info >> 5) & 1; }
  uint8_t getFDEType() const { return (sfde_func_info >> 4) & 1; }
  uint8_t getFREType() const { return sfde_func_info & 0xf; }
  void setPAuthKey(uint8_t P) { setFuncInfo(P, getFDEType(), getFREType()); }
  void setFDEType(uint8_t D) { setFuncInfo(getPAuthKey(), D, getFREType()); }
  void setFREType(uint8_t R) { setFuncInfo(getPAuthKey(), getFDEType(), R); }
  void setFuncInfo(uint8_t PAuthKey, uint8_t FDEType, uint8_t FREType) {
    sfde_func_info =
        ((PAuthKey & 1) << 5) | ((FDEType & 1) << 4) | (FREType & 0xf);
  }
};

struct sframe_fre_info {
  uint8_t fre_info;

  bool isReturnAddressSigned() const { return fre_info >> 7; }
  uint8_t getOffsetSize() const { return (fre_info >> 5) & 3; }
  uint8_t getOffsetCount() const { return (fre_info >> 1) & 0xf; }
  uint8_t getBaseRegister() const { return fre_info & 1; }
  void setReturnAddressSigned(bool RA) {
    setFREInfo(RA, getOffsetSize(), getOffsetCount(), getBaseRegister());
  }
  void setOffsetSize(uint8_t Sz) {
    setFREInfo(isReturnAddressSigned(), Sz, getOffsetCount(),
               getBaseRegister());
  }
  void setOffsetCount(uint8_t N) {
    setFREInfo(isReturnAddressSigned(), getOffsetSize(), N, getBaseRegister());
  }
  void setBaseRegister(uint8_t Reg) {
    setFREInfo(isReturnAddressSigned(), getOffsetSize(), getOffsetCount(), Reg);
  }
  void setFREInfo(bool RA, uint8_t Sz, uint8_t N, uint8_t Reg) {
    fre_info = ((RA & 1) << 7) | ((Sz & 3) << 5) | ((N & 0xf) << 1) | (Reg & 1);
  }
};

struct sframe_frame_row_entry_addr1 {
  uint8_t sfre_start_address;
  sframe_fre_info sfre_info;
};

struct sframe_frame_row_entry_addr2 {
  uint16_t sfre_start_address;
  sframe_fre_info sfre_info;
};

struct sframe_frame_row_entry_addr4 {
  uint32_t sfre_start_address;
  sframe_fre_info sfre_info;
};

LLVM_PACKED_END

} // namespace sframe
} // namespace llvm

#endif // LLVM_BINARYFORMAT_SFRAME_H
