//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSDKernel.h"

#include <cstddef>
#if defined(__FreeBSD__) && defined(__aarch64__)
#include <machine/frame.h>
#endif

using namespace lldb_private::platform_freebsdkernel;

// DWARF register numbers for AArch64.
static constexpr uint32_t kDwarfSP = 31;
static constexpr uint32_t kDwarfPC = 32;
static constexpr uint32_t kDwarfLR = 30;   // x30
static constexpr uint32_t kDwarfCPSR = 33; // PSTATE / CPSR

// Trapframe byte offsets.
// struct trapframe layout (sys/arm64/include/frame.h, FreeBSD 14+):
static constexpr int32_t kTfSP = 0;
static constexpr int32_t kTfLR = 8;
static constexpr int32_t kTfELR = 16;    // saved PC
static constexpr int32_t kTfSPSR = 24;   // cpsr
static constexpr int32_t kTfX0Base = 48; // tf_x[0]; tf_x[n] = kTfX0Base + n*8

#if defined(__FreeBSD__) && defined(__aarch64__)
static_assert(offsetof(struct trapframe, tf_sp) == (size_t)kTfSP,
              "tf_sp offset mismatch");
static_assert(offsetof(struct trapframe, tf_lr) == (size_t)kTfLR,
              "tf_lr offset mismatch");
static_assert(offsetof(struct trapframe, tf_elr) == (size_t)kTfELR,
              "tf_elr offset mismatch");
static_assert(offsetof(struct trapframe, tf_spsr) == (size_t)kTfSPSR,
              "tf_spsr offset mismatch");
static_assert(offsetof(struct trapframe, tf_x[0]) == (size_t)kTfX0Base,
              "tf_x[0] offset mismatch");
#endif

lldb::UnwindPlanSP PlatformFreeBSDKernel::GetTrapframeUnwindPlan_arm64(
    [[maybe_unused]] ConstString name) {
  std::vector<std::pair<uint32_t, int32_t>> regs;
  regs.reserve(33);

  regs.push_back({kDwarfSP, kTfSP});
  regs.push_back({kDwarfLR, kTfLR});
  regs.push_back({kDwarfPC, kTfELR});
  regs.push_back({kDwarfCPSR, kTfSPSR});

  for (uint32_t i = 0; i <= 29; i++)
    regs.push_back({i, kTfX0Base + i * 8});

  return BuildTrapframeUnwindPlan("FreeBSD aarch64 trapframe", kDwarfSP, 0,
                                  regs);
}
