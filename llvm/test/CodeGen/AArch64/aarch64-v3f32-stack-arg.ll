; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; Make sure that lowering a vector argument passed after the eight vector
; argument registers does not assert while reassembling the stack part.
define <3 x float> @v3f32_stack_arg(<3 x i32> %a0, <3 x i32> %a1,
                                    <3 x i32> %a2, <3 x i32> %a3,
                                    <3 x i32> %a4, <3 x i32> %a5,
                                    <3 x i32> %a6, <3 x i32> %a7,
                                    <3 x float> %a8) {
; CHECK-LABEL: v3f32_stack_arg:
; CHECK:       ldr d0, [sp]
; CHECK-NEXT:  orr x8, x8, #0x8
; CHECK-NEXT:  ld1 { v0.d }[1], [x8]
; CHECK-NEXT:  ret
  ret <3 x float> %a8
}
