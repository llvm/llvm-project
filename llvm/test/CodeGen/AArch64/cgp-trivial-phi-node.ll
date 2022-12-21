; Checks that case when GEP is bound to trivial PHI node is correctly handled.
; RUN: opt %s -mtriple=aarch64-linux-gnu -codegenprepare -S -o - | FileCheck %s

; CHECK:      define void @crash(ptr %s, i32 %n) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %struct = load ptr, ptr %s
; CHECK-NEXT:   %gep0 = getelementptr [65536 x i32], ptr %struct, i64 0, i32 20000
; CHECK-NEXT:   store i32 %n, ptr %gep0
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @crash(ptr %s, i32 %n) {
entry:
  %struct = load ptr, ptr %s
  %cmp = icmp slt i32 0, %n
  br i1 %cmp, label %baz, label %bar
baz:
  br label %bar

foo:
  %gep0 = getelementptr [65536 x i32], ptr %phi2, i64 0, i32 20000
  br label %st

st:
  store i32 %n, ptr %gep0
  br label %out

bar:
  %phi2 = phi ptr [ %struct, %baz ], [ %struct, %entry ]
  br label %foo
out:
  ret void
}
