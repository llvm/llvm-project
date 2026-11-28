; RUN: llc -O0 -mtriple thumbv7-windows-itanium -filetype asm -o - %s | FileCheck %s
; RUN: llc -O0 -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s
; RUN: llc -O0 -mtriple thumbv7-windows-mingw32 -filetype asm -o - %s | FileCheck %s

declare arm_aapcs_vfpcc i32 @num_entries()

define arm_aapcs_vfpcc void @test___builtin_alloca() {
entry:
  %array = alloca ptr, align 4
  %call = call arm_aapcs_vfpcc i32 @num_entries()
  %mul = mul i32 4, %call
  %0 = alloca i8, i32 %mul
  store ptr %0, ptr %array, align 4
  ret void
}

; CHECK: bl num_entries
; Any register is actually valid here, but turns out we use lr,
; because we do not have the kill flag on R0.
; CHECK: mov [[R0:r[0-9]+]], r0
; CHECK: movs [[R1:r[0-9]+]], #7
; CHECK: add.w [[R2:r[0-9]+]], [[R1]], [[R0]], lsl #2
; CHECK: bic [[R2]], [[R2]], #4
; CHECK: lsrs r4, [[R2]], #2
; CHECK: bl __chkstk
; CHECK: sub.w sp, sp, r4

; __chkstk only moves SP down by the size, so align it afterwards.
define arm_aapcs_vfpcc void @test_aligned_alloca() {
entry:
  %array = alloca ptr, align 4
  %call = call arm_aapcs_vfpcc i32 @num_entries()
  %mul = mul i32 4, %call
  %0 = alloca i8, i32 %mul, align 32
  store ptr %0, ptr %array, align 4
  ret void
}

; CHECK-LABEL: test_aligned_alloca:
; CHECK: bl __chkstk
; CHECK-NEXT: sub.w sp, sp, r4
; CHECK-NEXT: mov [[SP:r[0-9]+]], sp
; CHECK-NEXT: bic [[SP]], [[SP]], #31
; CHECK-NEXT: mov sp, [[SP]]

