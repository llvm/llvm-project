; RUN: llc < %s -stop-after=finalize-isel -O0 -march=nvptx64| FileCheck %s

define dso_local i32 @foo(i32 noundef %i, i32 noundef %b) #0 {
; CHECK-LABEL: body:
entry:
  %cmp = icmp eq i32 %i, 4
  br i1 %cmp, label %bb.1, label %bb.2
; CHECK: [[COND:%[0-9]+]]:int1regs = SETP_s32ri
; CHECK: Bra  %bb.2, killed [[COND]], 0
; CHECK: Jump %bb.1, $noreg, 0

bb.1:
; CHECK-LABEL: bb.1:
  %add = add nsw i32 %i, %b
  br label %bb.2
; CHECK: Jump %bb.2, $noreg, 0

bb.2:
; CHECK-LABEL: bb.2:
  %ret = phi i32 [%add, %bb.1], [%i, %entry]
  ret i32 %ret
; CHECK: Ret $noreg, 0
}
