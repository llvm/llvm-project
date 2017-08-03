; RUN: llc < %s -march=x86

define void @foobar() {
entry:
  %sub.i = trunc i64 undef to i32
  %shr80.i = ashr i32 %sub.i, 16
  %add82.i = add nsw i32 %shr80.i, 1
  %notlhs.i = icmp slt i32 %shr80.i, undef
  %notrhs.i = icmp sgt i32 %add82.i, -1
  %or.cond.not.i = and i1 %notrhs.i, %notlhs.i
  %cmp154.i = icmp slt i32 0, undef
  %or.cond406.i = and i1 %or.cond.not.i, %cmp154.i
  %or.cond406.not.i = xor i1 %or.cond406.i, true
  %or.cond407.i = or i1 undef, %or.cond406.not.i
  br i1 %or.cond407.i, label %if.then158.i, label %if.end163.i

if.then158.i:
  ret void

if.end163.i:                                      ; preds = %if.end67.i
  ret void
}
