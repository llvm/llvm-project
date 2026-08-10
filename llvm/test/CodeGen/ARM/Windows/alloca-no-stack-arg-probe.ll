; RUN: llc -mtriple thumbv7-windows -filetype asm -o - %s | FileCheck %s

declare arm_aapcs_vfpcc i32 @num_entries()

define arm_aapcs_vfpcc void @test___builtin_alloca() "no-stack-arg-probe" {
entry:
  %array = alloca ptr, align 4
  %call = call arm_aapcs_vfpcc i32 @num_entries()
  %mul = mul i32 4, %call
  %0 = alloca i8, i32 %mul
  store ptr %0, ptr %array, align 4
  ret void
}

; CHECK: bl num_entries
; CHECK: movs [[R1:r[0-9]+]], #7
; CHECK: add.w [[R0:r[0-9]+]], [[R1]], [[R0]], lsl #2
; CHECK: bic [[R0]], [[R0]], #7
; CHECK-NOT: bl __chkstk
; CHECK: sub.w [[R0]], sp, [[R0]]
; CHECK: mov sp, [[R0]]
