; RUN: llc -mtriple=aarch64--linux-gnu < %s | FileCheck %s

define dso_local void @_Z3fooi(i32 noundef %val) #0 {
; CHECK-LABEL: _Z3fooi:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub	sp, sp, #144
; CHECK-NEXT:    .cfi_def_cfa_offset 144
; CHECK-NEXT:    movi	v4.2d, #0xffffffff00000000
; CHECK-NEXT:    str	w0, [sp, #140]
; CHECK-NEXT:    adrp	x8, .LCPI0_0
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    ldp	q6, q0, [sp, #32]
; CHECK-NEXT:    ldr	q5, [x8, :lo12:.LCPI0_0]
; CHECK-NEXT:    stp	q1, q2, [sp, #64]
; CHECK-NEXT:    mov	v3.16b, v4.16b
; CHECK-NEXT:    stp	q5, q4, [sp]
; CHECK-NEXT:    bsl	v3.16b, v0.16b, v2.16b
; CHECK-NEXT:    mov	v0.16b, v5.16b
; CHECK-NEXT:    bsl	v0.16b, v6.16b, v1.16b
; CHECK-NEXT:    mov	v1.16b, v3.16b
; CHECK-NEXT:    stp	q0, q3, [sp, #96]
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    add	sp, sp, #144
; CHECK-NEXT:    ret
entry:
  %val.addr = alloca i32, align 4
  %x = alloca <8 x i32>, align 16
  %y = alloca <8 x i32>, align 16
  %z = alloca <8 x i32>, align 16
  %.compoundliteral = alloca <8 x i32>, align 16
  store i32 %val, ptr %val.addr, align 4
  %0 = call <8 x i32> asm sideeffect "", "={z1}"()
  store <8 x i32> %0, ptr %y, align 16
  store <8 x i32> <i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 -1, i32 0, i32 -1>, ptr %.compoundliteral, align 16
  %10 = load <8 x i32>, ptr %.compoundliteral, align 16
  %11 = load <8 x i32>, ptr %z, align 16
  %12 = load <8 x i32>, ptr %y, align 16
  %vector_cond = icmp ne <8 x i32> %10, zeroinitializer
  %vector_select = select <8 x i1> %vector_cond, <8 x i32> %11, <8 x i32> %12
  store <8 x i32> %vector_select, ptr %x, align 16
  %13 = load <8 x i32>, ptr %x, align 16
  call void asm sideeffect "", "{z0}"(<8 x i32> %13)
  ret void
}