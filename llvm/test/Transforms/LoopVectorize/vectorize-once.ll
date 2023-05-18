; RUN: opt < %s -passes=loop-vectorize,dce,instcombine,simplifycfg -force-vector-interleave=1 -force-vector-width=4 -S -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;
; We want to make sure that we are vectorizeing the scalar loop only once
; even if the pass manager runs the vectorizer multiple times due to inlining.


; This test checks that we add metadata to vectorized loops
; CHECK-LABEL: @_Z4foo1Pii(
; CHECK: <4 x i32>
; CHECK: br i1 %{{.*}}, label %{{.*}}, label %vector.body, !llvm.loop [[VEC_LOOP1:![0-9]+]]
;
; CHECK: br i1 %{{.*}}, label %{{.*}}, label %for.body.i, !llvm.loop [[SCALAR_LOOP1:![0-9]+]]
; CHECK: ret

; This test comes from the loop:
;
;int foo (int *A, int n) {
;  return std::accumulate(A, A + n, 0);
;}
define i32 @_Z4foo1Pii(ptr %A, i32 %n) #0 {
entry:
  %idx.ext = sext i32 %n to i64
  %add.ptr = getelementptr inbounds i32, ptr %A, i64 %idx.ext
  %cmp3.i = icmp eq i32 %n, 0
  br i1 %cmp3.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i

for.body.i:                                       ; preds = %entry, %for.body.i
  %__init.addr.05.i = phi i32 [ %add.i, %for.body.i ], [ 0, %entry ]
  %__first.addr.04.i = phi ptr [ %incdec.ptr.i, %for.body.i ], [ %A, %entry ]
  %0 = load i32, ptr %__first.addr.04.i, align 4
  %add.i = add nsw i32 %0, %__init.addr.05.i
  %incdec.ptr.i = getelementptr inbounds i32, ptr %__first.addr.04.i, i64 1
  %cmp.i = icmp eq ptr %incdec.ptr.i, %add.ptr
  br i1 %cmp.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %for.body.i, %entry
  %__init.addr.0.lcssa.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  ret i32 %__init.addr.0.lcssa.i
}

; This test checks that we don't vectorize loops that are marked with the "width" == 1 metadata.
; CHECK-LABEL: @_Z4foo2Pii(
; CHECK-NOT: <4 x i32>
; CHECK: br i1 %{{.*}}, label %{{.*}}, label %for.body.i, !llvm.loop [[SCALAR_LOOP2:![0-9]+]]
; CHECK: ret
define i32 @_Z4foo2Pii(ptr %A, i32 %n) #0 {
entry:
  %idx.ext = sext i32 %n to i64
  %add.ptr = getelementptr inbounds i32, ptr %A, i64 %idx.ext
  %cmp3.i = icmp eq i32 %n, 0
  br i1 %cmp3.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i

for.body.i:                                       ; preds = %entry, %for.body.i
  %__init.addr.05.i = phi i32 [ %add.i, %for.body.i ], [ 0, %entry ]
  %__first.addr.04.i = phi ptr [ %incdec.ptr.i, %for.body.i ], [ %A, %entry ]
  %0 = load i32, ptr %__first.addr.04.i, align 4
  %add.i = add nsw i32 %0, %__init.addr.05.i
  %incdec.ptr.i = getelementptr inbounds i32, ptr %__first.addr.04.i, i64 1
  %cmp.i = icmp eq ptr %incdec.ptr.i, %add.ptr
  br i1 %cmp.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i, !llvm.loop !0

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %for.body.i, %entry
  %__init.addr.0.lcssa.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  ret i32 %__init.addr.0.lcssa.i
}

attributes #0 = { nounwind readonly ssp uwtable "fp-contract-model"="standard" "frame-pointer"="non-leaf" "realign-stack" "relocation-model"="pic" "ssp-buffers-size"="8" }

; CHECK:      [[VEC_LOOP1]] = distinct !{[[VEC_LOOP1]], [[MD_IS_VEC:![0-9]+]], [[MD_RT_UNROLL_DIS:![0-9]+]]}
; CHECK-NEXT: [[MD_IS_VEC:![0-9]+]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK-NEXT: [[MD_RT_UNROLL_DIS]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK-NEXT: [[SCALAR_LOOP1]] = distinct !{[[SCALAR_LOOP1]], [[MD_RT_UNROLL_DIS:![0-9]+]], [[MD_IS_VEC]]}
; CHECK-NEXT: [[SCALAR_LOOP2]] = distinct !{[[SCALAR_LOOP2]], [[VEC_WIDTH_1:![0-9]+]]}
; CHECK-NEXT: [[VEC_WIDTH_1]] = !{!"llvm.loop.vectorize.width", i32 1}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", i32 1}
