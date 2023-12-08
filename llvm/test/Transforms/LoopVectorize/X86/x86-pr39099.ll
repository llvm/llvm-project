; RUN: opt -mcpu=skx -S -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

; This test checks the fix for PR39099.
;
; Check that the predicated load is not vectorized as an
; interleaved-group (which requires proper masking, currently unsupported)
; but rather as a scalarized accesses.
; (For SKX, Gather is not supported by the compiler for chars, therefore
;  the only remaining alternative is to scalarize).
;
; void masked_strided(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard) {
;   for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char t = p[2*ix];
;         q[ix] = t;
;     }
;   }
; }

;CHECK-LABEL: @masked_strided(
;CHECK: vector.body:
;CHECK-NEXT:  %index = phi i32
;CHECK-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;CHECK-NEXT:  %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;CHECK-NEXT:  %{{.*}} = shl nuw nsw <8 x i32> %[[VECIND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;CHECK-NEXT:  %[[M:.+]] = extractelement <8 x i1> %[[VMASK]], i32 0
;CHECK-NEXT:  br i1 %[[M]], label %pred.store.if, label %pred.store.continue
;CHECK-NOT:   %{{.+}} = load <16 x i8>, ptr %{{.*}}, align 1

define dso_local void @masked_strided(ptr noalias nocapture readonly %p, ptr noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.09 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.09, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.09, 1
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %mul
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, ptr %q, i32 %ix.09
  store i8 %0, ptr %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
