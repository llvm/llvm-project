; RUN: opt < %s -analyze -scalar-evolution -scalar-evolution-max-iterations=0 | FileCheck %s
; RUN: opt < %s -passes='print<scalar-evolution>' -disable-output 2>&1 | FileCheck %s
; PR1101

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = weak global [1000 x i32] zeroinitializer, align 32         

; CHECK-LABEL: Determining loop execution counts for: @test1
; CHECK: backedge-taken count is 10000

define void @test1(i32 %N) {
entry:
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32], [1000 x i32]* @A, i32 0, i32 %i.0          ; <i32*> [#uses=1]
        store i32 123, i32* %tmp
        %tmp2 = add i32 %i.0, 1         ; <i32> [#uses=1]
        br label %bb3

bb3:            ; preds = %bb, %entry
        %i.0 = phi i32 [ 0, %entry ], [ %tmp2, %bb ]            ; <i32> [#uses=3]
        %tmp3 = icmp sle i32 %i.0, 9999          ; <i1> [#uses=1]
        br i1 %tmp3, label %bb, label %bb5

bb5:            ; preds = %bb3
        br label %return

return:         ; preds = %bb5
        ret void
}

; PR22795
; CHECK-LABEL: Classifying expressions for: @test2
; CHECK:   %iv = phi i32 [ -1, %entry ], [ %next.1, %for.inc.1 ]
; CHECK-NEXT:  -->  {-1,+,2}<%preheader> U: full-set S: full-set             Exits: 13

define i32 @test2() {
entry:
  %bins = alloca [16 x i64], align 16
  %0 = bitcast [16 x i64]* %bins to i8*
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 128, i1 false)
  br label %preheader

preheader:                                        ; preds = %for.inc.1, %entry
  %v11 = phi i64 [ 0, %entry ], [ %next12.1, %for.inc.1 ]
  %iv = phi i32 [ -1, %entry ], [ %next.1, %for.inc.1 ]
  %cmp = icmp sgt i64 %v11, 0
  br i1 %cmp, label %for.body, label %for.inc

for.body:                                         ; preds = %preheader
  %zext = zext i32 %iv to i64
  %arrayidx = getelementptr [16 x i64], [16 x i64]* %bins, i64 0, i64 %v11
  %loaded = load i64, i64* %arrayidx, align 8
  %add = add i64 %loaded, 1
  %add2 = add i64 %add, %zext
  store i64 %add2, i64* %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body, %preheader
  %next12 = add nuw nsw i64 %v11, 1
  %next = add nsw i32 %iv, 1
  br i1 true, label %for.body.1, label %for.inc.1

end:                                              ; preds = %for.inc.1
  %arrayidx8 = getelementptr [16 x i64], [16 x i64]* %bins, i64 0, i64 2
  %load = load i64, i64* %arrayidx8, align 16
  %shr4 = lshr i64 %load, 32
  %conv = trunc i64 %shr4 to i32
  ret i32 %conv

for.body.1:                                       ; preds = %for.inc
  %zext.1 = zext i32 %next to i64
  %arrayidx.1 = getelementptr [16 x i64], [16 x i64]* %bins, i64 0, i64 %next12
  %loaded.1 = load i64, i64* %arrayidx.1, align 8
  %add.1 = add i64 %loaded.1, 1
  %add2.1 = add i64 %add.1, %zext.1
  store i64 %add2.1, i64* %arrayidx.1, align 8
  br label %for.inc.1

for.inc.1:                                        ; preds = %for.body.1, %for.inc
  %next12.1 = add nuw nsw i64 %next12, 1
  %next.1 = add nuw nsw i32 %next, 1
  %exitcond.1 = icmp eq i64 %next12.1, 16
  br i1 %exitcond.1, label %end, label %preheader
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #0
