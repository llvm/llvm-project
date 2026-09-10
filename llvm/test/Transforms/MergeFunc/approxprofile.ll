; RUN: opt -S -passes=mergefunc < %s | FileCheck %s
;
; Mixed approxprofile / exact pairs still merge. The lexicographically
; smaller name is the unmarked survivor, MergeFunctions ORs approxprofile
; onto it from @b_approx.

define i32 @a_exact(i32 %x) unnamed_addr {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 1
  %c = add i32 %b, 1
  %d = add i32 %c, 1
  ret i32 %d
}

define i32 @b_approx(i32 %x) unnamed_addr approxprofile {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 1
  %c = add i32 %b, 1
  %d = add i32 %c, 1
  ret i32 %d
}

; CHECK: define i32 @a_exact(i32 %x) unnamed_addr #[[A:[0-9]+]] {
; CHECK: define i32 @b_approx(i32 %{{.*}}) unnamed_addr #[[A]] {
; CHECK-NEXT: tail call i32 @a_exact
; CHECK: attributes #[[A]] = { approxprofile }
