; RUN: opt %loadPolly -polly-print-ast -polly-process-unprofitable -polly-allow-nonaffine -disable-output < %s | FileCheck %s
;
; @test1
; Make sure we generate the correct aliasing check for a fixed-size memset operation.
; CHECK: if (1 && (&MemRef_tmp0[15] <= &MemRef_tmp1[0] || &MemRef_tmp1[32] <= &MemRef_tmp0[14]))
;
; @test2
; Make sure we generate the correct aliasing check for a variable-size memset operation.
; CHECK: if (1 && (&MemRef_tmp0[15] <= &MemRef_tmp1[0] || &MemRef_tmp1[n] <= &MemRef_tmp0[14]))
;
; @test3
; We can't do anything interesting with a non-affine memset; just make sure it doesn't crash.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.info = type { i32, %struct.ctr*, i32, %struct.ord*, %struct.ctr*, i32, i8*, i32, i32, double }
%struct.ctr = type { i32, i8, i8, i32 }
%struct.ord = type { i32, i8 }

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #0

define void @test1(%struct.info** %ppIdxInfo) {
entry:
  %tmp0 = load %struct.info*, %struct.info** %ppIdxInfo, align 8
  br label %if.end125

if.end125:                                        ; preds = %entry
  %tmp1 = load %struct.ctr*, %struct.ctr** undef, align 8
  br label %for.end143

for.end143:                                       ; preds = %if.end125
  %tmp2 = bitcast %struct.ctr* %tmp1 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %tmp2, i8 0, i64 32, i32 4, i1 false)
  %needToFreeIdxStr = getelementptr inbounds %struct.info, %struct.info* %tmp0, i64 0, i32 7
  %tmp3 = load i32, i32* %needToFreeIdxStr, align 8
  br i1 false, label %if.end149, label %if.then148

if.then148:                                       ; preds = %for.end143
  br label %if.end149

if.end149:                                        ; preds = %if.then148, %for.end143
  ret void
}

define void @test2(%struct.info** %ppIdxInfo, i64 %n) {
entry:
  %tmp0 = load %struct.info*, %struct.info** %ppIdxInfo, align 8
  br label %if.end125

if.end125:                                        ; preds = %entry
  %tmp1 = load %struct.ctr*, %struct.ctr** undef, align 8
  br label %for.end143

for.end143:                                       ; preds = %if.end125
  %tmp2 = bitcast %struct.ctr* %tmp1 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %tmp2, i8 0, i64 %n, i32 4, i1 false)
  %needToFreeIdxStr = getelementptr inbounds %struct.info, %struct.info* %tmp0, i64 0, i32 7
  %tmp3 = load i32, i32* %needToFreeIdxStr, align 8
  br i1 false, label %if.end149, label %if.then148

if.then148:                                       ; preds = %for.end143
  br label %if.end149

if.end149:                                        ; preds = %if.then148, %for.end143
  ret void
}

define i32 @test3(i32* %x, i32 %n) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %conv = sext i32 %n to i64
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry.split
  %tmp0 = bitcast i32* %x to i8*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry.split
  ret i32 0

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.09 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul = mul nsw i64 %i.09, %i.09
  tail call void @llvm.memset.p0i8.i64(i8* %tmp0, i8 0, i64 %mul, i32 4, i1 false)
  %add = add nuw nsw i64 %i.09, 1000
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %add
  store i32 5, i32* %arrayidx, align 4
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond = icmp eq i64 %inc, %conv
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { argmemonly nounwind }
