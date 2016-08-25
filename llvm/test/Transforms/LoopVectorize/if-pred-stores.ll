; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=1 -force-vector-interleave=2 -loop-vectorize -verify-loop-info -simplifycfg < %s | FileCheck %s --check-prefix=UNROLL
; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=1 -force-vector-interleave=2 -loop-vectorize -verify-loop-info < %s | FileCheck %s --check-prefix=UNROLL-NOSIMPLIFY
; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=2 -force-vector-interleave=1 -loop-vectorize -enable-cond-stores-vec -verify-loop-info -simplifycfg < %s | FileCheck %s --check-prefix=VEC

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Test predication of stores.
define i32 @test(i32* nocapture %f) #0 {
entry:
  br label %for.body

; VEC-LABEL: test
; VEC:   %[[v0:.+]] = add i64 %index, 0
; VEC:   %[[v1:.+]] = add i64 %index, 1
; VEC:   %[[v2:.+]] = getelementptr inbounds i32, i32* %f, i64 %[[v0]]
; VEC:   %[[v4:.+]] = getelementptr inbounds i32, i32* %f, i64 %[[v1]]
; VEC:   %[[v8:.+]] = icmp sgt <2 x i32> %{{.*}}, <i32 100, i32 100>
; VEC:   %[[v9:.+]] = add nsw <2 x i32> %{{.*}}, <i32 20, i32 20>
; VEC:   %[[v10:.+]] = and <2 x i1> %[[v8]], <i1 true, i1 true>
; VEC:   %[[v11:.+]] = extractelement <2 x i1> %[[v10]], i32 0
; VEC:   %[[v12:.+]] = icmp eq i1 %[[v11]], true
; VEC:   br i1 %[[v12]], label %[[cond:.+]], label %[[else:.+]]
;
; VEC: [[cond]]:
; VEC:   %[[v13:.+]] = extractelement <2 x i32> %[[v9]], i32 0
; VEC:   store i32 %[[v13]], i32* %[[v2]], align 4
; VEC:   br label %[[else:.+]]
;
; VEC: [[else]]:
; VEC:   %[[v15:.+]] = extractelement <2 x i1> %[[v10]], i32 1
; VEC:   %[[v16:.+]] = icmp eq i1 %[[v15]], true
; VEC:   br i1 %[[v16]], label %[[cond2:.+]], label %[[else2:.+]]
;
; VEC: [[cond2]]:
; VEC:   %[[v17:.+]] = extractelement <2 x i32> %[[v9]], i32 1
; VEC:   store i32 %[[v17]], i32* %[[v4]], align 4
; VEC:   br label %[[else2:.+]]
;
; VEC: [[else2]]:

; UNROLL-LABEL: test
; UNROLL: vector.body:
; UNROLL:   %[[IND:[a-zA-Z0-9]+]] = add i64 %{{.*}}, 0
; UNROLL:   %[[IND1:[a-zA-Z0-9]+]] = add i64 %{{.*}}, 1
; UNROLL:   %[[v0:[a-zA-Z0-9]+]] = getelementptr inbounds i32, i32* %f, i64 %[[IND]]
; UNROLL:   %[[v1:[a-zA-Z0-9]+]] = getelementptr inbounds i32, i32* %f, i64 %[[IND1]]
; UNROLL:   %[[v2:[a-zA-Z0-9]+]] = load i32, i32* %[[v0]], align 4
; UNROLL:   %[[v3:[a-zA-Z0-9]+]] = load i32, i32* %[[v1]], align 4
; UNROLL:   %[[v4:[a-zA-Z0-9]+]] = icmp sgt i32 %[[v2]], 100
; UNROLL:   %[[v5:[a-zA-Z0-9]+]] = icmp sgt i32 %[[v3]], 100
; UNROLL:   %[[v6:[a-zA-Z0-9]+]] = add nsw i32 %[[v2]], 20
; UNROLL:   %[[v7:[a-zA-Z0-9]+]] = add nsw i32 %[[v3]], 20
; UNROLL:   %[[v8:[a-zA-Z0-9]+]] = icmp eq i1 %[[v4]], true
; UNROLL:   br i1 %[[v8]], label %[[cond:[a-zA-Z0-9.]+]], label %[[else:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond]]:
; UNROLL:   store i32 %[[v6]], i32* %[[v0]], align 4
; UNROLL:   br label %[[else]]
;
; UNROLL: [[else]]:
; UNROLL:   %[[v9:[a-zA-Z0-9]+]] = icmp eq i1 %[[v5]], true
; UNROLL:   br i1 %[[v9]], label %[[cond2:[a-zA-Z0-9.]+]], label %[[else2:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond2]]:
; UNROLL:   store i32 %[[v7]], i32* %[[v1]], align 4
; UNROLL:   br label %[[else2]]
;
; UNROLL: [[else2]]:

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %f, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 100
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %add = add nsw i32 %0, 20
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}

; Track basic blocks when unrolling conditional blocks. This code used to assert
; because we did not update the phi nodes with the proper predecessor in the
; vectorized loop body.
; PR18724

; UNROLL-NOSIMPLIFY-LABEL: bug18724
; UNROLL-NOSIMPLIFY: store i32
; UNROLL-NOSIMPLIFY: store i32

define void @bug18724() {
entry:
  br label %for.body9

for.body9:
  br i1 undef, label %for.inc26, label %for.body14

for.body14:
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc23 ], [ undef, %for.body9 ]
  %iNewChunks.120 = phi i32 [ %iNewChunks.2, %for.inc23 ], [ undef, %for.body9 ]
  %arrayidx16 = getelementptr inbounds [768 x i32], [768 x i32]* undef, i64 0, i64 %indvars.iv3
  %tmp = load i32, i32* %arrayidx16, align 4
  br i1 undef, label %if.then18, label %for.inc23

if.then18:
  store i32 2, i32* %arrayidx16, align 4
  %inc21 = add nsw i32 %iNewChunks.120, 1
  br label %for.inc23

for.inc23:
  %iNewChunks.2 = phi i32 [ %inc21, %if.then18 ], [ %iNewChunks.120, %for.body14 ]
  %indvars.iv.next4 = add nsw i64 %indvars.iv3, 1
  %tmp1 = trunc i64 %indvars.iv3 to i32
  %cmp13 = icmp slt i32 %tmp1, 0
  br i1 %cmp13, label %for.body14, label %for.inc26

for.inc26:
  %iNewChunks.1.lcssa = phi i32 [ undef, %for.body9 ], [ %iNewChunks.2, %for.inc23 ]
  unreachable
}
