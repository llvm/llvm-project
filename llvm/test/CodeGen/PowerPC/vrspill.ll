; RUN: llc -O0 -mtriple=powerpc-unknown-linux-gnu -mattr=+altivec -verify-machineinstrs  < %s | FileCheck %s
; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu -mattr=+altivec -verify-machineinstrs < %s | FileCheck %s

; This verifies that we generate correct spill/reload code for vector regs.

define void @addrtaken(i32 %i, <4 x float> %w) nounwind {
entry:
  %i.addr = alloca i32, align 4
  %w.addr = alloca <4 x float>, align 16
  store i32 %i, i32* %i.addr, align 4
  store <4 x float> %w, <4 x float>* %w.addr, align 16
  call void @foo(i32* %i.addr)
  ret void
}

; CHECK: stvx 2, 0, 0
; CHECK: lvx 2, 0, 0

declare void @foo(i32*)
