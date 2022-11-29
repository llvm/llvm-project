; RUN: llc -filetype=asm %s -o - -mtriple x86_64-apple-darwin | FileCheck %s
; RUN: llc -filetype=obj %s -o %t -mtriple x86_64-apple-darwin
; RUN: llvm-readobj --cg-profile %t | FileCheck %s --check-prefix=OBJ

declare void @b()

define void @a() {
  call void @b()
  ret void
}

define void @freq(i1 %cond) {
  br i1 %cond, label %A, label %B
A:
  call void @a();
  ret void
B:
  call void @b();
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 5, !"CG Profile", !1}
!1 = !{!2, !3, !4, !5}
!2 = !{ptr @a, ptr @b, i64 32}
!3 = !{ptr @freq, ptr @a, i64 11}
!4 = !{ptr @freq, ptr @b, i64 20}
!5 = !{ptr @freq, null, i64 20}

; CHECK: .cg_profile _a, _b, 32
; CHECK: .cg_profile _freq, _a, 11
; CHECK: .cg_profile _freq, _b, 20

; OBJ: CGProfile [
; OBJ:  CGProfileEntry {
; OBJ:    From: _a
; OBJ:    To: _b
; OBJ:    Weight: 32
; OBJ:  }
; OBJ:  CGProfileEntry {
; OBJ:    From: _freq
; OBJ:    To: _a
; OBJ:    Weight: 11
; OBJ:  }
; OBJ:  CGProfileEntry {
; OBJ:    From: _freq
; OBJ:    To: _b
; OBJ:    Weight: 20
; OBJ:  }
; OBJ:]
