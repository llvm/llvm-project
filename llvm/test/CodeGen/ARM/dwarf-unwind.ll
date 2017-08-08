; RUN: llc -mtriple=thumbv7-netbsd-eabi -o - %s | FileCheck %s
declare void @bar()

; ARM's frame lowering attempts to tack another callee-saved register onto the
; list when it detects a potential misaligned VFP store. However, if there are
; none available it used to just vpush anyway and misreport the location of the
; registers in unwind info. Since there are benefits to aligned stores, it's
; better to correct the code than the .cfi_offset directive.

define void @test_dpr_align(i8 %l, i8 %r) {
; CHECK-LABEL: test_dpr_align:
; CHECK: push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK: .cfi_def_cfa_offset 36
; CHECK: sub sp, #4
; CHECK: .cfi_def_cfa_offset 40
; CHECK: vpush {d8}
; CHECK: .cfi_offset d8, -48
; CHECK-NOT: sub sp
; [...]
; CHECK: bl bar
; CHECK-NOT: add sp
; CHECK: vpop {d8}
; CHECK: add sp, #4
; CHECK: pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{d8}"()
  call void @bar()
  ret void
}

; The prologue (but not the epilogue) can be made more space efficient by
; chucking an argument register into the list. Not worth it in general though,
; "sub sp, #4" is likely faster.
define void @test_dpr_align_tiny(i8 %l, i8 %r) minsize {
; CHECK-LABEL: test_dpr_align_tiny:
; CHECK: push.w {r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NOT: sub sp
; CHECK: vpush {d8}
; CHECK: .cfi_offset d8, -48
; CHECK-NOT: sub sp
; [...]
; CHECK: bl bar
; CHECK-NOT: add sp
; CHECK: vpop {d8}
; CHECK: add sp, #4
; CHECK: pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{d8}"()
  call void @bar()
  ret void
}


; However, we shouldn't do a 2-step align/adjust if there are no DPRs to be
; saved.
define void @test_nodpr_noalign(i8 %l, i8 %r) {
; CHECK-LABEL: test_nodpr_noalign:
; CHECK: push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NOT: sub sp
; CHECK: sub sp, #12
; CHECK-NOT: sub sp
; [...]
; CHECK: bl bar
; CHECK-NOT: add sp
; CHECK: add sp, #12
; CHECK-NOT: add sp
; CHECK: pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
  alloca i64
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"()
  call void @bar()
  ret void
}

define void @test_frame_pointer_offset() minsize "no-frame-pointer-elim"="true" {
; CHECK-LABEL: test_frame_pointer_offset:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: .cfi_def_cfa_offset 20
; CHECK: add r7, sp, #12
; CHECK: .cfi_def_cfa r7, 8
; CHECK-NOT: .cfi_def_cfa_offset
; CHECK: push.w {r7, r8, r9, r10, r11}
; CHECK-NOT: .cfi_def_cfa_offset
  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{d8}"()
  call void @bar()
  ret void
}
