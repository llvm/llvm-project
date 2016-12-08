; RUN: opt < %s -mattr=avx -force-vector-width=2 -force-vector-interleave=1 -loop-vectorize -simplifycfg -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: predicated_sdiv_masked_load
;
; This test ensures that we don't scalarize the predicated load. Since the load
; can be vectorized with predication, scalarizing it would cause its pointer
; operand to become non-uniform.
;
; CHECK: vector.body:
; CHECK:   %wide.masked.load = call <2 x i32> @llvm.masked.load.v2i32.p0v2i32
; CHECK:   br i1 {{.*}}, label %[[IF0:.+]], label %[[CONT0:.+]]
; CHECK: [[IF0]]:
; CHECK:   %[[T0:.+]] = extractelement <2 x i32> %wide.masked.load, i32 0
; CHECK:   %[[T1:.+]] = sdiv i32 %[[T0]], %x
; CHECK:   %[[T2:.+]] = insertelement <2 x i32> undef, i32 %[[T1]], i32 0
; CHECK:   br label %[[CONT0]]
; CHECK: [[CONT0]]:
; CHECK:   %[[T3:.+]] = phi <2 x i32> [ undef, %vector.body ], [ %[[T2]], %[[IF0]] ]
; CHECK:   br i1 {{.*}}, label %[[IF1:.+]], label %[[CONT1:.+]]
; CHECK: [[IF1]]:
; CHECK:   %[[T4:.+]] = extractelement <2 x i32> %wide.masked.load, i32 1
; CHECK:   %[[T5:.+]] = sdiv i32 %[[T4]], %x
; CHECK:   %[[T6:.+]] = insertelement <2 x i32> %[[T3]], i32 %[[T5]], i32 1
; CHECK:   br label %[[CONT1]]
; CHECK: [[CONT1]]:
; CHECK:   phi <2 x i32> [ %[[T3]], %[[CONT0]] ], [ %[[T6]], %[[IF1]] ]
; CHECK:   br i1 {{.*}}, label %middle.block, label %vector.body

define i32 @predicated_sdiv_masked_load(i32* %a, i32* %b, i32 %x, i1 %c) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i32 [ 0, %entry ], [ %tmp7, %for.inc ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp2 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp3 = load i32, i32* %tmp2, align 4
  %tmp4 = sdiv i32 %tmp3, %x
  %tmp5 = add nsw i32 %tmp4, %tmp1
  br label %for.inc

for.inc:
  %tmp6 = phi i32 [ %tmp1, %for.body ], [ %tmp5, %if.then]
  %tmp7 = add i32 %r, %tmp6
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 10000
  br i1 %cond, label %for.end, label %for.body

for.end:
  %tmp8 = phi i32 [ %tmp7, %for.inc ]
  ret i32 %tmp8
}
