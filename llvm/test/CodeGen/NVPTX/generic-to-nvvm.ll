; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx-nvidia-cuda"

; Ensure global variables in address space 0 are promoted to address space 1

; CHECK: .global .align 4 .u32 myglobal = 42;
@myglobal = internal global i32 42, align 4
; CHECK: .global .align 4 .u32 myconst = 420;
@myconst = internal constant i32 420, align 4


define void @foo(i32* %a, i32* %b) {
; Expect one load -- @myconst isn't loaded from, because we know its value
; statically.
; CHECK: ld.global.u32
; CHECK: st.global.u32
; CHECK: st.global.u32
  %ld1 = load i32, i32* @myglobal
  %ld2 = load i32, i32* @myconst
  store i32 %ld1, i32* %a
  store i32 %ld2, i32* %b
  ret void
}


!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*)* @foo, !"kernel", i32 1}
