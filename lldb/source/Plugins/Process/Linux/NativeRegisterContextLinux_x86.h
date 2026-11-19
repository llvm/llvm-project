//===-- NativeRegisterContextLinux_x86.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__i386__) || defined(__x86_64__)

#ifndef lldb_NativeRegisterContextLinux_x86_h
#define lldb_NativeRegisterContextLinux_x86_h

#include "Plugins/Process/Linux/NativeRegisterContextLinux.h"
#include "Plugins/Register/Native/NativeRegisterContextDBReg_x86.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include <optional>
#include <sys/uio.h>

namespace lldb_private {
namespace process_linux {

LLVM_PACKED_START
struct XSAVE_HDR {
  enum class XFeature : uint64_t {
    FP = 1,
    SSE = FP << 1,
    YMM = SSE << 1,
    BNDREGS = YMM << 1,
    BNDCSR = BNDREGS << 1,
    OPMASK = BNDCSR << 1,
    ZMM_Hi256 = OPMASK << 1,
    Hi16_ZMM = ZMM_Hi256 << 1,
    PT = Hi16_ZMM << 1,
    PKRU = PT << 1,
    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ PKRU)
  };

  XFeature xstate_bv; // OS enabled xstate mask to determine the extended states
                      // supported by the processor
  XFeature xcomp_bv;  // Mask to indicate the format of the XSAVE area and of
                      // the XRSTOR instruction
  uint64_t reserved1[1];
  uint64_t reserved2[5];
};

static_assert(sizeof(XSAVE_HDR) == 64, "XSAVE_HDR layout incorrect");
LLVM_PACKED_END

// Fixed XSAVE fields accessed by the Linux backend.
LLVM_PACKED_START
struct XSAVE {
  // Legacy floating-point and SIMD state.
  RegisterLayout_x86::FXSAVE i387;
  XSAVE_HDR header;

  // Upper halves of the YMM registers; lower halves are in i387.xmm.
  RegisterLayout_x86::YMMHReg ymmh[16];
  uint64_t reserved3[16];

  // MPX bounds and control/status state.
  RegisterLayout_x86::MPXReg mpxr[4];
  RegisterLayout_x86::MPXCsr mpxc[2];
};
LLVM_PACKED_END

// Floating-point registers
union FPR {
  RegisterLayout_x86::FXSAVE fxsave; // Generic floating-point registers.
  XSAVE xsave;                       // x86 extended processor state.
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// External LLDB numbering used by Linux register access, not logical IDs.
namespace x86_64_regnum {

enum : uint32_t {
  // General Purpose Registers.
  k_first_gpr,
  rax = k_first_gpr,
  rbx,
  rcx,
  rdx,
  rdi,
  rsi,
  rbp,
  rsp,
  r8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  rip,
  rflags,

  // Segment Registers.
  cs,
  fs,
  gs,
  ss,

  // Segment Base Registers.
  fs_base,
  gs_base,

  // Segment Registers.
  ds,
  es,

  // Subregister Aliases.
  k_first_alias,
  eax = k_first_alias,
  ebx,
  ecx,
  edx,
  edi,
  esi,
  ebp,
  esp,
  r8d,  // Low 32 bits of r8
  r9d,  // Low 32 bits of r9
  r10d, // Low 32 bits of r10
  r11d, // Low 32 bits of r11
  r12d, // Low 32 bits of r12
  r13d, // Low 32 bits of r13
  r14d, // Low 32 bits of r14
  r15d, // Low 32 bits of r15
  ax,
  bx,
  cx,
  dx,
  di,
  si,
  bp,
  sp,
  r8w,  // Low 16 bits of r8
  r9w,  // Low 16 bits of r9
  r10w, // Low 16 bits of r10
  r11w, // Low 16 bits of r11
  r12w, // Low 16 bits of r12
  r13w, // Low 16 bits of r13
  r14w, // Low 16 bits of r14
  r15w, // Low 16 bits of r15
  ah,
  bh,
  ch,
  dh,
  al,
  bl,
  cl,
  dl,
  dil,
  sil,
  bpl,
  spl,
  r8l,  // Low 8 bits of r8
  r9l,  // Low 8 bits of r9
  r10l, // Low 8 bits of r10
  r11l, // Low 8 bits of r11
  r12l, // Low 8 bits of r12
  r13l, // Low 8 bits of r13
  r14l, // Low 8 bits of r14
  r15l, // Low 8 bits of r15
  k_last_alias = r15l,

  k_last_gpr = k_last_alias,

  // Floating Point Registers.
  k_first_fpr,

  // Floating Point Control and Status Registers.
  fctrl = k_first_fpr,
  fstat,
  ftag,
  fop,
  fiseg,
  fioff,
  fip,
  foseg,
  fooff,
  fdp,

  // SSE Control and Status Registers.
  mxcsr,
  mxcsrmask,

  // Floating Point Stack Registers.
  st0,
  st1,
  st2,
  st3,
  st4,
  st5,
  st6,
  st7,

  // MMX Registers.
  mm0,
  mm1,
  mm2,
  mm3,
  mm4,
  mm5,
  mm6,
  mm7,

  // SSE Vector Registers.
  xmm0,
  xmm1,
  xmm2,
  xmm3,
  xmm4,
  xmm5,
  xmm6,
  xmm7,
  xmm8,
  xmm9,
  xmm10,
  xmm11,
  xmm12,
  xmm13,
  xmm14,
  xmm15,
  k_last_fpr = xmm15,

  // AVX Vector Registers.
  k_first_avx,
  ymm0 = k_first_avx,
  ymm1,
  ymm2,
  ymm3,
  ymm4,
  ymm5,
  ymm6,
  ymm7,
  ymm8,
  ymm9,
  ymm10,
  ymm11,
  ymm12,
  ymm13,
  ymm14,
  ymm15,
  k_last_avx = ymm15,

  // MPX Bound Registers.
  k_first_mpxr,
  bnd0 = k_first_mpxr,
  bnd1,
  bnd2,
  bnd3,
  k_last_mpxr = bnd3,

  // MPX Control Registers.
  k_first_mpxc,
  bndcfgu = k_first_mpxc,
  bndstatus,
  k_last_mpxc = bndstatus,

  // Debug Registers.
  k_first_dbr,
  dr0 = k_first_dbr,
  dr1,
  dr2,
  dr3,
  dr4,
  dr5,
  dr6,
  dr7,
  k_last_dbr = dr7,

  // Register Counts.
  k_num_registers,
  k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
  k_num_fpr_registers = k_last_fpr - k_first_fpr + 1,
  k_num_avx_registers = k_last_avx - k_first_avx + 1,
  k_num_mpx_registers = k_last_mpxc - k_first_mpxr + 1,
  k_num_user_registers = k_num_gpr_registers + k_num_fpr_registers +
      k_num_avx_registers + k_num_mpx_registers,
  k_num_dbr_registers = k_last_dbr - k_first_dbr + 1,
};

} // namespace x86_64_regnum


class NativeProcessLinux;

class NativeRegisterContextLinux_x86
    : public NativeRegisterContextLinux,
      public NativeRegisterContextDBReg_x86 {
public:
  NativeRegisterContextLinux_x86(const ArchSpec &target_arch,
                                    NativeThreadProtocol &native_thread);

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) override;

  Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) override;

  std::optional<SyscallData> GetSyscallData() override;

  std::optional<MmapData> GetMmapData() override;

protected:
  void *GetGPRBuffer() override { return &m_gpr_x86_64; }

  void *GetFPRBuffer() override;

  size_t GetFPRSize() override;

  Status ReadFPR() override;

  Status WriteFPR() override;

  uint32_t GetPtraceOffset(uint32_t reg_index) override;

private:
  // Private member types.
  enum class XStateType { Invalid, FXSAVE, XSAVE };
  enum class RegSet { gpr, fpu, avx, mpx };

  // Info about register ranges.
  struct RegInfo {
    uint32_t num_registers;
    uint32_t num_gpr_registers;
    uint32_t num_fpr_registers;
    uint32_t num_avx_registers;
    uint32_t num_mpx_registers;
    uint32_t last_gpr;
    uint32_t first_fpr;
    uint32_t last_fpr;
    uint32_t first_st;
    uint32_t last_st;
    uint32_t first_mm;
    uint32_t last_mm;
    uint32_t first_xmm;
    uint32_t last_xmm;
    uint32_t first_ymm;
    uint32_t last_ymm;
    uint32_t first_mpxr;
    uint32_t last_mpxr;
    uint32_t first_mpxc;
    uint32_t last_mpxc;
    uint32_t first_dr;
    uint32_t last_dr;
    uint32_t gpr_flags;
  };

  // Private member variables.
  mutable XStateType m_xstate_type;
  std::unique_ptr<FPR, llvm::FreeDeleter>
      m_xstate; // Extended States Area, named FPR for historical reasons.
  struct iovec m_iovec;
  RegInfo m_reg_info;
  struct {
    const char *name = nullptr;
    uint32_t byte_offset = 0;
  } m_orig_ax;
  uint64_t m_gpr_x86_64[x86_64_regnum::k_num_gpr_registers];

  // Private member methods.
  bool IsCPUFeatureAvailable(RegSet feature_code) const;

  bool IsRegisterSetAvailable(uint32_t set_index) const;

  bool IsGPR(uint32_t reg_index) const;

  bool IsFPR(uint32_t reg_index) const;

  bool IsDR(uint32_t reg_index) const;

  bool IsAVX(uint32_t reg_index) const;

  bool IsMPX(uint32_t reg_index) const;

  bool IsThreadPointer(const RegisterDescriptor &reg_info) const;

  Status ReadThreadPointer(RegisterValue &reg_value);

  void UpdateXSTATEforWrite(uint32_t reg_index);
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef lldb_NativeRegisterContextLinux_x86_h

#endif // defined(__i386__) || defined(__x86_64__)
