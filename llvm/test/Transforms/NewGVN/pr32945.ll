; RUN: opt -S -newgvn %s | FileCheck %s
; CHECK-NOT: call i32 @llvm.ssa.copy

@d = external global i32
@e = external global i32
define void @tinkywinky() {
  br i1 true, label %lor.lhs.false, label %cond.true
lor.lhs.false:
  %tmp = load i32, i32* @d, align 4
  %patatino = load i32, i32* null, align 4
  %or = or i32 %tmp, %patatino
  store i32 %or, i32* @d, align 4
  br label %cond.true
cond.true:
  %tmp1 = load i32, i32* @e, align 4
  %tmp2 = load i32, i32* @d, align 4
  %cmp = icmp eq i32 %tmp1, %tmp2
  br i1 %cmp, label %cond.true6, label %cond.false
cond.true6:
  %cmp7 = icmp slt i32 %tmp1, 0
  br i1 %cmp7, label %cond.false, label %cond.false
cond.false:
  ret void
}
