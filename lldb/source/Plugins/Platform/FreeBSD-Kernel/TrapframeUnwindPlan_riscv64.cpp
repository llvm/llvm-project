//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSDKernel.h"

#include <cstddef>
#if defined(__FreeBSD__) && defined(__riscv)
#include <machine/frame.h>
#endif

using namespace lldb_private::platform_freebsdkernel;

// DWARF register numbers for RISC-V.
static constexpr uint32_t kDwarfSP = 2;  // x2
static constexpr uint32_t kDwarfPC = 32; // pc

// struct trapframe layout (sys/riscv/include/frame.h):
#if defined(__FreeBSD__) && defined(__riscv)
static_assert(offsetof(struct trapframe, tf_ra) == (size_t)(0 * 8),
              "tf_ra offset mismatch");
static_assert(offsetof(struct trapframe, tf_sp) == (size_t)(1 * 8),
              "tf_sp offset mismatch");
static_assert(offsetof(struct trapframe, tf_gp) == (size_t)(2 * 8),
              "tf_gp offset mismatch");
static_assert(offsetof(struct trapframe, tf_tp) == (size_t)(3 * 8),
              "tf_tp offset mismatch");
static_assert(offsetof(struct trapframe, tf_t[0]) == (size_t)(4 * 8),
              "tf_t[0] offset mismatch");
static_assert(offsetof(struct trapframe, tf_t[3]) == (size_t)(7 * 8),
              "tf_t[3] offset mismatch");
static_assert(offsetof(struct trapframe, tf_s[0]) == (size_t)(11 * 8),
              "tf_s[0] offset mismatch");
static_assert(offsetof(struct trapframe, tf_a[0]) == (size_t)(23 * 8),
              "tf_a[0] offset mismatch");
static_assert(offsetof(struct trapframe, tf_sepc) == (size_t)(31 * 8),
              "tf_sepc offset mismatch");
#endif

lldb::UnwindPlanSP PlatformFreeBSDKernel::GetTrapframeUnwindPlan_riscv64(
    [[maybe_unused]] ConstString name) {
  return BuildTrapframeUnwindPlan(
      "FreeBSD riscv64 trapframe", kDwarfSP, 0,
      {
          // slot 0–3: ra, sp, gp, tp
          {1, 0 * 8}, // ra  (x1)
          {2, 1 * 8}, // sp  (x2) — caller's sp saved here
          {3, 2 * 8}, // gp  (x3)
          {4, 3 * 8}, // tp  (x4)
                      // slot 4–6: t0–t2
          {5, 4 * 8}, // t0  (x5)
          {6, 5 * 8}, // t1  (x6)
          {7,
           6 * 8}, // t2  (x7)
                   // slot 7–10: t3–t6  ← BEFORE s0/s1; matches FreeBSD frame.h
          {28, 7 * 8},  // t3  (x28)
          {29, 8 * 8},  // t4  (x29)
          {30, 9 * 8},  // t5  (x30)
          {31, 10 * 8}, // t6  (x31)
                        // slot 11–12: s0 (fp), s1
          {8, 11 * 8},  // s0/fp (x8)
          {9, 12 * 8},  // s1    (x9)
                        // slot 13–22: s2–s11
          {18, 13 * 8}, // s2  (x18)
          {19, 14 * 8}, // s3  (x19)
          {20, 15 * 8}, // s4  (x20)
          {21, 16 * 8}, // s5  (x21)
          {22, 17 * 8}, // s6  (x22)
          {23, 18 * 8}, // s7  (x23)
          {24, 19 * 8}, // s8  (x24)
          {25, 20 * 8}, // s9  (x25)
          {26, 21 * 8}, // s10 (x26)
          {27, 22 * 8}, // s11 (x27)
                        // slot 23–30: a0–a7
          {10, 23 * 8}, // a0  (x10)
          {11, 24 * 8}, // a1  (x11)
          {12, 25 * 8}, // a2  (x12)
          {13, 26 * 8}, // a3  (x13)
          {14, 27 * 8}, // a4  (x14)
          {15, 28 * 8}, // a5  (x15)
          {16, 29 * 8}, // a6  (x16)
          {17, 30 * 8}, // a7  (x17)
                        // slot 31: sepc = saved PC
          {kDwarfPC, 31 * 8},
      });
}
