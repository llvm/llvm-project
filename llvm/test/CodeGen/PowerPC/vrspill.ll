; RUN: llc -O0 -mtriple=powerpc-unknown-linux-gnu -mattr=+altivec -mattr=-vsx -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu -mattr=+altivec -mattr=-vsx -verify-machineinstrs -fast-isel=false -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu -mattr=+altivec -mattr=+vsx -verify-machineinstrs -fast-isel=false -mcpu=pwr7 < %s | FileCheck -check-prefix=CHECK-VSX %s

; This verifies that we generate correct spill/reload code for vector regs.

define void @addrtaken(i32 %i, <4 x float> %w) nounwind {
entry:
  %i.addr = alloca i32, align 4
  %w.addr = alloca <4 x float>, align 16
  store i32 %i, ptr %i.addr, align 4
  store <4 x float> %w, ptr %w.addr, align 16
  call void @foo(ptr %i.addr)
  ret void
}

; CHECK: stvx 2,

; We would prefer to test for "stxvw4x 34," but current -O0 code
; needlessly generates "vor 3,2,2 / stxvw4x 35,0,3", so we'll settle for
; the opcode.
; CHECK-VSX: stxvw4x

declare void @foo(ptr)
