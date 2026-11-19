; RUN: llc -mtriple=x86_64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -verify-machineinstrs -regalloc=basic < %s | FileCheck %s

; The statepoint has more live-in deopt values than there are registers, so
; some of them must be spilled and folded into the statepoint.

@g0 = external global i8
@g1 = external global i8
@g2 = external global i8
@g3 = external global i8
@g4 = external global i8
@g5 = external global i8
@g6 = external global i8
@g7 = external global i8
@g8 = external global i8
@g9 = external global i8
@g10 = external global i8
@g11 = external global i8
@g12 = external global i8
@g13 = external global i8
@g14 = external global i8
@g15 = external global i8

; CHECK-LABEL: test:
; CHECK: callq foo{{.*}}Folded Reload
define void @test() gc "statepoint-example" {
  %t = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 2, i32 0, i32 0) ["deopt"(ptr @g0, ptr @g1, ptr @g2, ptr @g3, ptr @g4, ptr @g5, ptr @g6, ptr @g7, ptr @g8, ptr @g9, ptr @g10, ptr @g11, ptr @g12, ptr @g13, ptr @g14, ptr @g15)]
  ret void
}

declare void @foo()
declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)
