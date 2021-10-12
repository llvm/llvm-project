; RUN: opt -S -globaldce < %s | FileCheck %s

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
@b = internal unnamed_addr constant i64* @c
@c = internal unnamed_addr constant i64 42
@d = internal unnamed_addr constant i64 42

; All four, and mainly @d need to stay alive:
; CHECK: @a = internal unnamed_addr constant i64 42
; CHECK: @b = internal unnamed_addr constant i64* @c
; CHECK: @c = internal unnamed_addr constant i64 42
; CHECK: @d = internal unnamed_addr constant i64 42

@llvm.used = appending global [3 x i8*] [
  i8* bitcast (i64* @a to i8*),
  i8* bitcast (i64** @b to i8*),
  i8* bitcast (i64* @d to i8*)
], section "llvm.metadata"

!1 = !{i64** @b, i32 0, !{i64* @a}}
!2 = !{i64* @d, i32 0, !{i64* @c}}
!llvm.used.conditional = !{!1, !2}
