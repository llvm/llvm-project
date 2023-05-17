; RUN: opt < %s -loop-reduce -S

; Test that SCEV insertpoint's don't get corrupted and cause an
; invalid instruction to be inserted in a block other than its parent.
; See http://reviews.llvm.org/D20703 for context.
define void @test() {
entry:
  %bf.load = load i32, ptr null, align 4
  %bf.clear = lshr i32 %bf.load, 1
  %div = and i32 %bf.clear, 134217727
  %sub = add nsw i32 %div, -1
  %0 = zext i32 %sub to i64
  br label %while.cond

while.cond:                                       ; preds = %cond.end, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %cond.end ], [ 0, %entry ]
  %cmp = icmp eq i64 %indvars.iv, %0
  br i1 %cmp, label %cleanup16, label %while.body

while.body:                                       ; preds = %while.cond
  %1 = trunc i64 %indvars.iv to i32
  %mul = shl i32 %1, 1
  %add = add nuw i32 %mul, 2
  %cmp3 = icmp ult i32 %add, 0
  br i1 %cmp3, label %if.end, label %if.then

if.then:                                          ; preds = %while.body
  unreachable

if.end:                                           ; preds = %while.body
  br i1 false, label %cond.end, label %cond.true

cond.true:                                        ; preds = %if.end
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %if.end
  %add7 = add i32 %1, 1
  %cmp12 = icmp ugt i32 %add7, %sub
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %cmp12, label %if.then13, label %while.cond

if.then13:                                        ; preds = %cond.end
  unreachable

cleanup16:                                        ; preds = %while.cond
  ret void
}
