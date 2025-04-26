; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc-unknown-unkown \
; RUN:   -mcpu=pwr7 -O0 2>&1 | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64-unknown-unkown \
; RUN:   -mcpu=pwr7 -O0 2>&1 | FileCheck %s

define void @test_r1_clobber() {
entry:
  call void asm sideeffect "nop", "~{r1}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: R1
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

define void @test_x1_clobber() {
entry:
  call void asm sideeffect "nop", "~{x1}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: X1
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

; CHECK: warning: inline asm clobber list contains reserved registers: R31
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.

@a = dso_local global i32 100, align 4
define dso_local signext i32 @main() {
entry:
  %retval = alloca i32, align 4
  %old = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  call void asm sideeffect "li 31, 1", "~{r31}"()
  call void asm sideeffect "li 30, 1", "~{r30}"()
  %0 = call i64 asm sideeffect "mr $0, 31", "=r"()
  store i64 %0, ptr %old, align 8
  %1 = load i32, ptr @a, align 4
  %conv = sext i32 %1 to i64
  %2 = alloca i8, i64 %conv, align 16
  %3 = load i64, ptr %old, align 8
  %conv1 = trunc i64 %3 to i32
  ret i32 %conv1
}
