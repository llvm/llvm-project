; RUN: opt -basic-aa -loop-idiom < %s -S | FileCheck %s

target datalayout = "e-p:32:32:32-p1:64:64:64-p2:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Two dimensional nested loop should be promoted to one big memset.
define void @test10(i8 addrspace(2)* %X) nounwind ssp {
; CHECK-LABEL: @test10(
; CHECK: entry:
; CHECK-NEXT: call void @llvm.memset.p2i8.i16(i8 addrspace(2)* align 1 %X, i8 0, i16 10000, i1 false)
; CHECK-NOT: store
; CHECK: ret void

entry:
  br label %bb.nph

bb.nph:                                           ; preds = %entry, %for.inc10
  %i.04 = phi i16 [ 0, %entry ], [ %inc12, %for.inc10 ]
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %bb.nph
  %j.02 = phi i16 [ 0, %bb.nph ], [ %inc, %for.body5 ]
  %mul = mul nsw i16 %i.04, 100
  %add = add nsw i16 %j.02, %mul
  %arrayidx = getelementptr inbounds i8, i8 addrspace(2)* %X, i16 %add
  store i8 0, i8 addrspace(2)* %arrayidx, align 1
  %inc = add nsw i16 %j.02, 1
  %cmp4 = icmp eq i16 %inc, 100
  br i1 %cmp4, label %for.inc10, label %for.body5

for.inc10:                                        ; preds = %for.body5
  %inc12 = add nsw i16 %i.04, 1
  %cmp = icmp eq i16 %inc12, 100
  br i1 %cmp, label %for.end13, label %bb.nph

for.end13:                                        ; preds = %for.inc10
  ret void
}

define void @test11_pattern(i32 addrspace(2)* nocapture %P) nounwind ssp {
; CHECK-LABEL: @test11_pattern(
; CHECK-NOT: memset_pattern
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32, i32 addrspace(2)* %P, i64 %indvar
  store i32 1, i32 addrspace(2)* %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; PR9815 - This is a partial overlap case that cannot be safely transformed
; into a memcpy.
@g_50 = addrspace(2) global [7 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0], align 16


define i32 @test14() nounwind {
; CHECK-LABEL: @test14(
; CHECK: for.body:
; CHECK: load i32
; CHECK: store i32
; CHECK: br i1 %cmp

entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %tmp5 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %add = add nsw i32 %tmp5, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds [7 x i32], [7 x i32] addrspace(2)* @g_50, i32 0, i64 %idxprom
  %tmp2 = load i32, i32 addrspace(2)* %arrayidx, align 4
  %add4 = add nsw i32 %tmp5, 5
  %idxprom5 = sext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds [7 x i32], [7 x i32] addrspace(2)* @g_50, i32 0, i64 %idxprom5
  store i32 %tmp2, i32 addrspace(2)* %arrayidx6, align 4
  %inc = add nsw i32 %tmp5, 1
  %cmp = icmp slt i32 %inc, 2
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  %tmp8 = load i32, i32 addrspace(2)* getelementptr inbounds ([7 x i32], [7 x i32] addrspace(2)* @g_50, i32 0, i64 6), align 4
  ret i32 %tmp8
}

