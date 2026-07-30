; RUN: llc -mtriple=aarch64-none-linux-gnu %s -o - -O1 -debug-only=consthoist 2>&1 | FileCheck %s
; REQUIRES: asserts

declare void @g(i64)

; Single ORR.
; CHECK:     Function: f1
; CHECK-NOT: Collect constant
define void @f1(i1 %cond) {
entry:
  call void @g(i64 -3)
  br i1 %cond, label %true, label %ret

true:
  call void @g(i64 -3)
  br label %ret

ret:
  ret void
}

; Constant is 0xBEEF000000000000, single MOVZ with shift.
; CHECK:     Function: f2
; CHECK-NOT: Collect constant
define void @f2(i1 %cond, i64 %p, i64 %q) {
entry:
  %a = and i64 %p, 13758215386640154624
  call void @g(i64 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i64 %q, 13758215386640154624
  call void @g(i64 %b)
  br label %ret

ret:
  ret void
}

; CHECK:     Function: f3
; CHECK:     Collect constant i64 4294967103 from   %a = and i64 %p, 4294967103 with cost 2
define void @f3(i1 %cond, i64 %p, i64 %q) {
entry:
  %a = and i64 %p, 4294967103
  call void @g(i64 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i64 %q, 4294967103
  call void @g(i64 %b)
  br label %ret

ret:
  ret void
}

; CHECK:     Function: f4
; Collect constant i64 -4688528683866062848 from   %a = and i64 %p, -4688528683866062848 with cost 2
define void @f4(i1 %cond, i64 %p, i64 %q) {
entry:
  %a = and i64 %p, 13758215389843488768
  call void @g(i64 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i64 %q, 13758215389843488768
  call void @g(i64 %b)
  br label %ret

ret:
  ret void
}

; CHECK:     Function: f5
; Collect constant i64 88994925642865 from   %a = and i64 %p, 88994925642865 with cost 3
define void @f5(i1 %cond, i64 %p, i64 %q) {
entry:
  %a = and i64 %p, 88994925642865
  call void @g(i64 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i64 %q, 88994925642865
  call void @g(i64 %b)
  br label %ret

ret:
  ret void
}

; CHECK:     Function: f6
; Collect constant i64 -4688439692143754127 from   %b = and i64 %q, -4688439692143754127 with cost 4
define void @f6(i1 %cond, i64 %p, i64 %q) {
entry:
  %a = and i64 %p, 13758304381565797489
  call void @g(i64 %a)
  br i1 %cond, label %true, label %ret

true:
  %b = and i64 %q, 13758304381565797489
  call void @g(i64 %b)
  br label %ret

ret:
  ret void
}
