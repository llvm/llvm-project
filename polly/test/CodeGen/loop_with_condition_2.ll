; RUN: opt %loadPolly -basic-aa -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s

; Verify that we actually detect this loop as the innermost loop even though
; there is a conditional inside.

; CHECK: #pragma simd
; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1) {
; CHECK:   if (c0 >= m + 1025) {
; CHECK:     Stmt_if_else(c0);
; CHECK:   } else
; CHECK:     Stmt_if_then(c0);
; CHECK:   Stmt_if_end(c0);
; CHECK: }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16

define void @loop_with_condition(i32 %m) nounwind {
entry:
  fence seq_cst
  %tmp = sub i32 0, %m
  %tmp1 = zext i32 %tmp to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr [1024 x i32], ptr @A, i64 0, i64 %indvar
  %arrayidx10 = getelementptr [1024 x i32], ptr @B, i64 0, i64 %indvar
  %tmp2 = add i64 %tmp1, %indvar
  %sub = trunc i64 %tmp2 to i32
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp3 = icmp sle i32 %sub, 1024
  br i1 %cmp3, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  store i32 1, ptr %arrayidx
  br label %if.end

if.else:                                          ; preds = %for.body
  store i32 2, ptr %arrayidx
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  store i32 3, ptr %arrayidx10
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvar.next = add i64 %indvar, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
entry:
  call void @llvm.memset.p0.i64(ptr @A, i8 0, i64 4096, i32 1, i1 false)
  call void @llvm.memset.p0.i64(ptr @B, i8 0, i64 4096, i32 1, i1 false)
  call void @loop_with_condition(i32 5)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr [1024 x i32], ptr @B, i64 0, i64 %indvar1
  %i.0 = trunc i64 %indvar1 to i32
  %cmp = icmp slt i32 %i.0, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp3 = load i32, ptr %arrayidx
  %cmp4 = icmp ne i32 %tmp3, 3
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %return

if.end:                                           ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvar.next2 = add i64 %indvar1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc12, %for.end
  %indvar = phi i64 [ %indvar.next, %for.inc12 ], [ 0, %for.end ]
  %arrayidx15 = getelementptr [1024 x i32], ptr @A, i64 0, i64 %indvar
  %i.1 = trunc i64 %indvar to i32
  %cmp8 = icmp slt i32 %i.1, 1024
  br i1 %cmp8, label %for.body9, label %for.end35

for.body9:                                        ; preds = %for.cond6
  br i1 true, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %for.body9
  %tmp16 = load i32, ptr %arrayidx15
  %cmp17 = icmp ne i32 %tmp16, 1
  br i1 %cmp17, label %if.then18, label %if.else

if.then18:                                        ; preds = %land.lhs.true
  br label %return

if.else:                                          ; preds = %land.lhs.true, %for.body9
  br i1 false, label %land.lhs.true23, label %if.end30

land.lhs.true23:                                  ; preds = %if.else
  %tmp27 = load i32, ptr %arrayidx15
  %cmp28 = icmp ne i32 %tmp27, 2
  br i1 %cmp28, label %if.then29, label %if.end30

if.then29:                                        ; preds = %land.lhs.true23
  br label %return

if.end30:                                         ; preds = %land.lhs.true23, %if.else
  br label %if.end31

if.end31:                                         ; preds = %if.end30
  br label %for.inc12

for.inc12:                                        ; preds = %if.end31
  %indvar.next = add i64 %indvar, 1
  br label %for.cond6

for.end35:                                        ; preds = %for.cond6
  br label %return

return:                                           ; preds = %for.end35, %if.then29, %if.then18, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 1, %if.then18 ], [ 1, %if.then29 ], [ 0, %for.end35 ]
  ret i32 %retval.0
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i32, i1) nounwind
