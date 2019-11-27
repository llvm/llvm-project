; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Make sure all regs are spilled
define anyregcc void @anyregcc1() {
entry:
;CHECK-LABEL: anyregcc1
;CHECK: stmg %r2, %r15, 16(%r15)
;CHECK: aghi %r15, -256
;CHECK: std %f0, 384(%r15)
;CHECK: std %f1,
;CHECK: std %f2, 392(%r15)
;CHECK: std %f3,
;CHECK: std %f4, 400(%r15)
;CHECK: std %f5,
;CHECK: std %f6, 408(%r15)
;CHECK: std %f7,
;CHECK: std %f8,
;CHECK: std %f9,
;CHECK: std %f10,
;CHECK: std %f11,
;CHECK: std %f12,
;CHECK: std %f13,
;CHECK: std %f14,
;CHECK: std %f15,
;CHECK: .cfi_offset %f0, -32
;CHECK: .cfi_offset %f2, -24
;CHECK: .cfi_offset %f4, -16
;CHECK: .cfi_offset %f6, -8
;CHECK: ld %f0, 384(%r15)
;CHECK: ld %f2, 392(%r15)
;CHECK: ld %f4, 400(%r15)
;CHECK: ld %f6, 408(%r15)
  call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15}"() nounwind
  ret void
}

; Make sure we don't spill any FPs
declare anyregcc void @foo()
define void @anyregcc2() {
entry:
;CHECK-LABEL: anyregcc2
;CHECK-NOT: std
;CHECK: std %f8,
;CHECK-NEXT: std %f9,
;CHECK-NEXT: std %f10,
;CHECK-NEXT: std %f11,
;CHECK-NEXT: std %f12,
;CHECK-NEXT: std %f13,
;CHECK-NEXT: std %f14,
;CHECK-NEXT: std %f15,
;CHECK-NOT: std
  %a0 = call double asm sideeffect "", "={f0}"() nounwind
  %a1 = call double asm sideeffect "", "={f1}"() nounwind
  %a2 = call double asm sideeffect "", "={f2}"() nounwind
  %a3 = call double asm sideeffect "", "={f3}"() nounwind
  %a4 = call double asm sideeffect "", "={f4}"() nounwind
  %a5 = call double asm sideeffect "", "={f5}"() nounwind
  %a6 = call double asm sideeffect "", "={f6}"() nounwind
  %a7 = call double asm sideeffect "", "={f7}"() nounwind
  %a8 = call double asm sideeffect "", "={f8}"() nounwind
  %a9 = call double asm sideeffect "", "={f9}"() nounwind
  %a10 = call double asm sideeffect "", "={f10}"() nounwind
  %a11 = call double asm sideeffect "", "={f11}"() nounwind
  %a12 = call double asm sideeffect "", "={f12}"() nounwind
  %a13 = call double asm sideeffect "", "={f13}"() nounwind
  %a14 = call double asm sideeffect "", "={f14}"() nounwind
  %a15 = call double asm sideeffect "", "={f15}"() nounwind
  call anyregcc void @foo()
  call void asm sideeffect "", "{f0},{f1},{f2},{f3},{f4},{f5},{f6},{f7},{f8},{f9},{f10},{f11},{f12},{f13},{f14},{f15}"(double %a0, double %a1, double %a2, double %a3, double %a4, double %a5, double %a6, double %a7, double %a8, double %a9, double %a10, double %a11, double %a12, double %a13, double %a14, double %a15)
  ret void
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
