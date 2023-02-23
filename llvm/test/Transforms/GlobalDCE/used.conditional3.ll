; RUN: opt -S -passes=globaldce < %s | FileCheck %s

; Test that when performing llvm.used.conditional removals, we discover globals
; that might trigger other llvm.used.conditional liveness. See the following
; diagram of dependencies between the globals:
;
; @a -\
;      \ (conditional, see !1, satisfied)
;       \-> @b -\
;                \ (regular usage)
;                 \-> @c -\
;                          \ (conditional, see !2, satisfied)
;                           \-> @d

@a = internal unnamed_addr constant i64 42
@b = internal unnamed_addr constant ptr @c
@c = internal unnamed_addr constant i64 42
@d = internal unnamed_addr constant i64 42

; All four, and mainly @d need to stay alive:
; CHECK: @a = internal unnamed_addr constant i64 42
; CHECK: @b = internal unnamed_addr constant ptr @c
; CHECK: @c = internal unnamed_addr constant i64 42
; CHECK: @d = internal unnamed_addr constant i64 42

@llvm.used = appending global [3 x ptr] [
  ptr @a,
  ptr @b,
  ptr @d
], section "llvm.metadata"

!1 = !{ptr @b, i32 0, !{ptr @a}}
!2 = !{ptr @d, i32 0, !{ptr @c}}
!llvm.used.conditional = !{!1, !2}
