//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSDKernel.h"

#include <cstddef>
#if defined(__FreeBSD__) && defined(__amd64__)
#include <machine/frame.h>
#endif

using namespace lldb_private::platform_freebsdkernel;

// DWARF register numbers for x86-64 (SYSV AMD64 ABI).
static constexpr uint32_t kDwarfRAX = 0;
static constexpr uint32_t kDwarfRDX = 1;
static constexpr uint32_t kDwarfRCX = 2;
static constexpr uint32_t kDwarfRBX = 3;
static constexpr uint32_t kDwarfRSI = 4;
static constexpr uint32_t kDwarfRDI = 5;
static constexpr uint32_t kDwarfRBP = 6;
static constexpr uint32_t kDwarfRSP = 7;
static constexpr uint32_t kDwarfR8 = 8;
static constexpr uint32_t kDwarfR9 = 9;
static constexpr uint32_t kDwarfR10 = 10;
static constexpr uint32_t kDwarfR11 = 11;
static constexpr uint32_t kDwarfR12 = 12;
static constexpr uint32_t kDwarfR13 = 13;
static constexpr uint32_t kDwarfR14 = 14;
static constexpr uint32_t kDwarfR15 = 15;
static constexpr uint32_t kDwarfRIP = 16;
static constexpr uint32_t kDwarfRFLAGS = 49;
static constexpr uint32_t kDwarfCS = 51;
static constexpr uint32_t kDwarfSS = 52;

// Trapframe byte offsets
//
// struct trapframe layout (sys/x86/include/frame.h):
static constexpr int32_t kTfRDI = 0 * 8;
static constexpr int32_t kTfRSI = 1 * 8;
static constexpr int32_t kTfRDX = 2 * 8;
static constexpr int32_t kTfRCX = 3 * 8;
static constexpr int32_t kTfR8 = 4 * 8;
static constexpr int32_t kTfR9 = 5 * 8;
static constexpr int32_t kTfRAX = 6 * 8;
static constexpr int32_t kTfRBX = 7 * 8;
static constexpr int32_t kTfRBP = 8 * 8;
static constexpr int32_t kTfR10 = 9 * 8;
static constexpr int32_t kTfR11 = 10 * 8;
static constexpr int32_t kTfR12 = 11 * 8;
static constexpr int32_t kTfR13 = 12 * 8;
static constexpr int32_t kTfR14 = 13 * 8;
static constexpr int32_t kTfR15 = 14 * 8;
static constexpr int32_t kTfRIP = 19 * 8;
static constexpr int32_t kTfCS = 20 * 8;
static constexpr int32_t kTfRFLAGS = 21 * 8;
static constexpr int32_t kTfRSP = 22 * 8;
static constexpr int32_t kTfSS = 23 * 8;

#if defined(__FreeBSD__) && defined(__amd64__)
static_assert(offsetof(struct trapframe, tf_rdi) == (size_t)kTfRDI,
              "tf_rdi offset mismatch");
static_assert(offsetof(struct trapframe, tf_rsi) == (size_t)kTfRSI,
              "tf_rsi offset mismatch");
static_assert(offsetof(struct trapframe, tf_rdx) == (size_t)kTfRDX,
              "tf_rdx offset mismatch");
static_assert(offsetof(struct trapframe, tf_rcx) == (size_t)kTfRCX,
              "tf_rcx offset mismatch");
static_assert(offsetof(struct trapframe, tf_r8) == (size_t)kTfR8,
              "tf_r8 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r9) == (size_t)kTfR9,
              "tf_r9 offset mismatch");
static_assert(offsetof(struct trapframe, tf_rax) == (size_t)kTfRAX,
              "tf_rax offset mismatch");
static_assert(offsetof(struct trapframe, tf_rbx) == (size_t)kTfRBX,
              "tf_rbx offset mismatch");
static_assert(offsetof(struct trapframe, tf_rbp) == (size_t)kTfRBP,
              "tf_rbp offset mismatch");
static_assert(offsetof(struct trapframe, tf_r10) == (size_t)kTfR10,
              "tf_r10 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r11) == (size_t)kTfR11,
              "tf_r11 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r12) == (size_t)kTfR12,
              "tf_r12 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r13) == (size_t)kTfR13,
              "tf_r13 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r14) == (size_t)kTfR14,
              "tf_r14 offset mismatch");
static_assert(offsetof(struct trapframe, tf_r15) == (size_t)kTfR15,
              "tf_r15 offset mismatch");
static_assert(offsetof(struct trapframe, tf_rip) == (size_t)kTfRIP,
              "tf_rip offset mismatch");
static_assert(offsetof(struct trapframe, tf_cs) == (size_t)kTfCS,
              "tf_cs offset mismatch");
static_assert(offsetof(struct trapframe, tf_rflags) == (size_t)kTfRFLAGS,
              "tf_rflags offset mismatch");
static_assert(offsetof(struct trapframe, tf_rsp) == (size_t)kTfRSP,
              "tf_rsp offset mismatch");
static_assert(offsetof(struct trapframe, tf_ss) == (size_t)kTfSS,
              "tf_ss offset mismatch");
#endif

lldb::UnwindPlanSP PlatformFreeBSDKernel::GetTrapframeUnwindPlan_x86_64(
    [[maybe_unused]] ConstString name) {
  return BuildTrapframeUnwindPlan(
      "FreeBSD amd64 trapframe", kDwarfRSP, 0,
      {
          {kDwarfRDI, kTfRDI}, {kDwarfRSI, kTfRSI}, {kDwarfRDX, kTfRDX},
          {kDwarfRCX, kTfRCX}, {kDwarfR8, kTfR8},   {kDwarfR9, kTfR9},
          {kDwarfRAX, kTfRAX}, {kDwarfRBX, kTfRBX}, {kDwarfRBP, kTfRBP},
          {kDwarfR10, kTfR10}, {kDwarfR11, kTfR11}, {kDwarfR12, kTfR12},
          {kDwarfR13, kTfR13}, {kDwarfR14, kTfR14}, {kDwarfR15, kTfR15},
          {kDwarfRIP, kTfRIP}, {kDwarfCS, kTfCS},   {kDwarfRFLAGS, kTfRFLAGS},
          {kDwarfRSP, kTfRSP}, {kDwarfSS, kTfSS},
      });
}
