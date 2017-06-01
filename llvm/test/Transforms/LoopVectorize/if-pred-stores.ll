; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=1 -force-vector-interleave=2 -loop-vectorize -verify-loop-info -simplifycfg < %s | FileCheck %s --check-prefix=UNROLL
; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=1 -force-vector-interleave=2 -loop-vectorize -verify-loop-info < %s | FileCheck %s --check-prefix=UNROLL-NOSIMPLIFY
; RUN: opt -S -vectorize-num-stores-pred=1 -force-vector-width=2 -force-vector-interleave=1 -loop-vectorize -verify-loop-info -simplifycfg < %s | FileCheck %s --check-prefix=VEC

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Test predication of stores.
define i32 @test(i32* nocapture %f) #0 {
entry:
  br label %for.body

; VEC-LABEL: test
; VEC:   %[[v0:.+]] = add i64 %index, 0
; VEC:   %[[v8:.+]] = icmp sgt <2 x i32> %{{.*}}, <i32 100, i32 100>
; VEC:   %[[v10:.+]] = and <2 x i1> %[[v8]], <i1 true, i1 true>
; VEC:   %[[o1:.+]] = or <2 x i1> zeroinitializer, %[[v10]]
; VEC:   %[[v11:.+]] = extractelement <2 x i1> %[[o1]], i32 0
; VEC:   %[[v12:.+]] = icmp eq i1 %[[v11]], true
; VEC:   br i1 %[[v12]], label %[[cond:.+]], label %[[else:.+]]
;
; VEC: [[cond]]:
; VEC:   %[[v13:.+]] = extractelement <2 x i32> %wide.load, i32 0
; VEC:   %[[v9a:.+]] = add nsw i32 %[[v13]], 20
; VEC:   %[[v2:.+]] = getelementptr inbounds i32, i32* %f, i64 %[[v0]]
; VEC:   store i32 %[[v9a]], i32* %[[v2]], align 4
; VEC:   br label %[[else:.+]]
;
; VEC: [[else]]:
; VEC:   %[[v15:.+]] = extractelement <2 x i1> %[[o1]], i32 1
; VEC:   %[[v16:.+]] = icmp eq i1 %[[v15]], true
; VEC:   br i1 %[[v16]], label %[[cond2:.+]], label %[[else2:.+]]
;
; VEC: [[cond2]]:
; VEC:   %[[v17:.+]] = extractelement <2 x i32> %wide.load, i32 1
; VEC:   %[[v9b:.+]] = add nsw i32 %[[v17]], 20
; VEC:   %[[v1:.+]] = add i64 %index, 1
; VEC:   %[[v4:.+]] = getelementptr inbounds i32, i32* %f, i64 %[[v1]]
; VEC:   store i32 %[[v9b]], i32* %[[v4]], align 4
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
; UNROLL:   %[[o1:[a-zA-Z0-9]+]] = or i1 false, %[[v4]]
; UNROLL:   %[[o2:[a-zA-Z0-9]+]] = or i1 false, %[[v5]]
; UNROLL:   %[[v8:[a-zA-Z0-9]+]] = icmp eq i1 %[[o1]], true
; UNROLL:   br i1 %[[v8]], label %[[cond:[a-zA-Z0-9.]+]], label %[[else:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond]]:
; UNROLL:   %[[v6:[a-zA-Z0-9]+]] = add nsw i32 %[[v2]], 20
; UNROLL:   store i32 %[[v6]], i32* %[[v0]], align 4
; UNROLL:   br label %[[else]]
;
; UNROLL: [[else]]:
; UNROLL:   %[[v9:[a-zA-Z0-9]+]] = icmp eq i1 %[[o2]], true
; UNROLL:   br i1 %[[v9]], label %[[cond2:[a-zA-Z0-9.]+]], label %[[else2:[a-zA-Z0-9.]+]]
;
; UNROLL: [[cond2]]:
; UNROLL:   %[[v7:[a-zA-Z0-9]+]] = add nsw i32 %[[v3]], 20
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

; VEC-LABEL: @minimal_bit_widths(
;
; In the test below, it's more profitable for the expression feeding the
; conditional store to remain scalar. Since we can only type-shrink vector
; types, we shouldn't try to represent the expression in a smaller type.
;
; VEC: vector.body:
; VEC:   %wide.load = load <2 x i8>, <2 x i8>* {{.*}}, align 1
; VEC:   br i1 {{.*}}, label %[[IF0:.+]], label %[[CONT0:.+]]
; VEC: [[IF0]]:
; VEC:   %[[E0:.+]] = extractelement <2 x i8> %wide.load, i32 0
; VEC:   %[[Z0:.+]] = zext i8 %[[E0]] to i32
; VEC:   %[[T0:.+]] = trunc i32 %[[Z0]] to i8
; VEC:   store i8 %[[T0]], i8* {{.*}}, align 1
; VEC:   br label %[[CONT0]]
; VEC: [[CONT0]]:
; VEC:   br i1 {{.*}}, label %[[IF1:.+]], label %[[CONT1:.+]]
; VEC: [[IF1]]:
; VEC:   %[[E1:.+]] = extractelement <2 x i8> %wide.load, i32 1
; VEC:   %[[Z1:.+]] = zext i8 %[[E1]] to i32
; VEC:   %[[T1:.+]] = trunc i32 %[[Z1]] to i8
; VEC:   store i8 %[[T1]], i8* {{.*}}, align 1
; VEC:   br label %[[CONT1]]
; VEC: [[CONT1]]:
; VEC:   br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @minimal_bit_widths(i1 %c) {
entry:
  br label %for.body

for.body:
  %tmp0 = phi i64 [ %tmp6, %for.inc ], [ 0, %entry ]
  %tmp1 = phi i64 [ %tmp7, %for.inc ], [ undef, %entry ]
  %tmp2 = getelementptr i8, i8* undef, i64 %tmp0
  %tmp3 = load i8, i8* %tmp2, align 1
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp4 = zext i8 %tmp3 to i32
  %tmp5 = trunc i32 %tmp4 to i8
  store i8 %tmp5, i8* %tmp2, align 1
  br label %for.inc

for.inc:
  %tmp6 = add nuw nsw i64 %tmp0, 1
  %tmp7 = add i64 %tmp1, -1
  %tmp8 = icmp eq i64 %tmp7, 0
  br i1 %tmp8, label %for.end, label %for.body

for.end:
  ret void
}
