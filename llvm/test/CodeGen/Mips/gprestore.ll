; DISABLE: llc -march=mips < %s | FileCheck %s
; RUN: false
; XFAIL: *

@p = external global i32
@q = external global i32
@r = external global i32

define void @f0() nounwind {
entry:
; CHECK: jalr
; CHECK-NOT: got({{.*}})($gp)
; CHECK: lw $gp
; CHECK: jalr
; CHECK-NOT: got({{.*}})($gp)
; CHECK: lw $gp
; CHECK: jalr
; CHECK-NOT: got({{.*}})($gp)
; CHECK: lw $gp
  tail call void (...) @f1() nounwind
  %tmp = load i32, i32* @p, align 4
  tail call void @f2(i32 %tmp) nounwind
  %tmp1 = load i32, i32* @q, align 4
  %tmp2 = load i32, i32* @r, align 4
  tail call void @f3(i32 %tmp1, i32 %tmp2) nounwind
  ret void
}

declare void @f1(...)

declare void @f2(i32)

declare void @f3(i32, i32)

