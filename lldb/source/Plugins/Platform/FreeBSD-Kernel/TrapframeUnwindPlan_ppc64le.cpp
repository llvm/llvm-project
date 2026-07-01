//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSDKernel.h"

#include <cstddef>
#if defined(__FreeBSD__) && defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
#include <machine/frame.h>
#endif

using namespace lldb_private::platform_freebsdkernel;

// DWARF register numbers for PowerPC64 ELF ABI v2.
static constexpr uint32_t kDwarfR1 = 1; // sp
static constexpr uint32_t kDwarfLR = 65;
static constexpr uint32_t kDwarfCTR = 66;
static constexpr uint32_t kDwarfCR = 68;
static constexpr uint32_t kDwarfXER = 76;

// Trapframe slot offsets.
//
// struct trapframe layout (from sys/powerpc/include/frame.h):
static constexpr int32_t kOffFixreg = 0 * 8;
static constexpr int32_t kOffLR = 32 * 8;
static constexpr int32_t kOffCR = 33 * 8;
static constexpr int32_t kOffXER = 34 * 8;
static constexpr int32_t kOffCTR = 35 * 8;
// kOffSRR0 = 36 * 8: saved PC
// SRR0 (the saved PC / return address from trap) has no standard DWARF
// register number in the PowerPC ABI.  LLDB maps it via the architecture
// plugin's gdbarch_pc_regnum equivalent, but there is no portable DWARF
// number to use here.  We therefore omit PC recovery from the UnwindPlan;
// LLDB will use LR as a fallback for the return address when PC is unknown,
// which is the correct behaviour for kernel trap frames where LR holds the
// pre-trap link register (not the PC).
//
// In practice: the frame above the trap handler will have an unknown PC,
// but all GP registers including r1 (sp) will be correctly recovered,
// which is sufficient for a useful backtrace.

// On PowerPC, SP does NOT point at the base of struct trapframe.
// The trapframe is above the standard linkage area on the stack:
//
//   base = SP + 48  (48-byte ABI linkage area)
static constexpr int32_t kPPC64LinkageArea = 48;

#if defined(__FreeBSD__) && defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
static_assert(offsetof(struct trapframe, fixreg[0]) ==
                  (size_t)(kOffFixreg + 0 * 8),
              "fixreg[0] offset mismatch");
static_assert(offsetof(struct trapframe, fixreg[31]) ==
                  (size_t)(kOffFixreg + 31 * 8),
              "fixreg[31] offset mismatch");
static_assert(offsetof(struct trapframe, lr) == (size_t)kOffLR,
              "lr offset mismatch");
static_assert(offsetof(struct trapframe, cr) == (size_t)kOffCR,
              "cr offset mismatch");
static_assert(offsetof(struct trapframe, xer) == (size_t)kOffXER,
              "xer offset mismatch");
static_assert(offsetof(struct trapframe, ctr) == (size_t)kOffCTR,
              "ctr offset mismatch");
#endif

lldb::UnwindPlanSP PlatformFreeBSDKernel::GetTrapframeUnwindPlan_ppc64le(
    [[maybe_unused]] ConstString name) {
  // CFA = r1 (SP) + 48.  All register offsets are relative to CFA
  // (= base of struct trapframe), matching kgdb's `base = SP + 48`.

  std::vector<std::pair<uint32_t, int32_t>> regs;
  regs.reserve(37);

  // r0–r31 from tf_fixreg[].
  for (uint32_t i = 0; i <= 31; i++)
    regs.push_back({i, kOffFixreg + i * 8});

  // Special registers.
  regs.push_back({kDwarfLR, kOffLR});
  regs.push_back({kDwarfCR, kOffCR});
  regs.push_back({kDwarfXER, kOffXER});
  regs.push_back({kDwarfCTR, kOffCTR});
  // SRR0 (PC) deliberately omitted — no standard DWARF register number.

  return BuildTrapframeUnwindPlan("FreeBSD ppc64le trapframe", kDwarfR1,
                                  kPPC64LinkageArea, regs);
}
