; RUN: opt -S -globaldce < %s | FileCheck %s

; Test !llvm.used.conditional with circular dependencies

@globalA = internal unnamed_addr constant i8* bitcast (i8** @globalB to i8*)
@globalB = internal unnamed_addr constant i8* bitcast (i8** @globalA to i8*)

; All four, and mainly @d need to stay alive:
; CHECK-NOT: @globalA
; CHECK-NOT: @globalB

@llvm.used = appending global [2 x i8*] [
  i8* bitcast (i8** @globalA to i8*),
  i8* bitcast (i8** @globalB to i8*)
], section "llvm.metadata"

!1 = !{i8** @globalA, i32 0, !{i8** @globalB}}
!2 = !{i8** @globalB, i32 0, !{i8** @globalA}}
!llvm.used.conditional = !{!1, !2}
