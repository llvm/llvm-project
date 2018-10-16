; RUN: opt -mcpu=skx -S -loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses < %s | FileCheck %s -check-prefix=DISABLED_MASKED_STRIDED 
; RUN: opt -mcpu=skx -S -loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -enable-interleaved-mem-accesses  -enable-masked-interleaved-mem-accesses < %s | FileCheck %s -check-prefix=ENABLED_MASKED_STRIDED 

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; When masked-interleaved-groups are disabled:
; Check that the predicated load is not vectorized as an
; interleaved-group but rather as a scalarized accesses.
; (For SKX, Gather is not supported by the compiler for chars, therefore
;  the only remaining alternative is to scalarize).
; When  masked-interleave-group is enabled we expect to find the proper mask
; shuffling code, feeding the wide masked load for an interleave-group (with
; a single member).
;
; void masked_strided1(const unsigned char* restrict p,
;                      unsigned char* restrict q,
;                      unsigned char guard) {
;   for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char t = p[2*ix];
;         q[ix] = t;
;     }
;   }
; }

;DISABLED_MASKED_STRIDED-LABEL: @masked_strided1(
;DISABLED_MASKED_STRIDED: vector.body:
;DISABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;DISABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %[[WIDEVEC]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;DISABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;DISABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shl nuw nsw <8 x i32> %[[VECIND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;DISABLED_MASKED_STRIDED-NEXT:  %[[M:.+]] = extractelement <8 x i1> %[[VMASK]], i32 0
;DISABLED_MASKED_STRIDED-NEXT:  br i1 %[[M]], label %pred.load.if, label %pred.load.continue
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>

;ENABLED_MASKED_STRIDED-LABEL: @masked_strided1(
;ENABLED_MASKED_STRIDED: vector.body:
;ENABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;ENABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;ENABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;ENABLED_MASKED_STRIDED:       %interleaved.mask = shufflevector <8 x i1> %[[VMASK]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
;ENABLED_MASKED_STRIDED-NEXT:  %[[WIDEMASKEDLOAD:.+]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask, <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:  %[[STRIDEDVEC:.+]] = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>

define dso_local void @masked_strided1(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.09 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.09, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.09, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %arrayidx3 = getelementptr inbounds i8, i8* %q, i32 %ix.09
  store i8 %0, i8* %arrayidx3, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.09, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Check also a scenario with full interleave-groups (no gaps) as well as both
; load and store groups. We check that when masked-interleave-group is disabled
; the predicated loads (and stores) are not vectorized as an
; interleaved-group but rather as four separate scalarized accesses.
; (For SKX, gather/scatter is not supported by the compiler for chars, therefore
; the only remaining alternative is to scalarize).
; When  masked-interleave-group is enabled we expect to find the proper mask
; shuffling code, feeding the wide masked load/store for the two interleave-
; groups.
;
; void masked_strided2(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard) {
;         char left = p[2*ix];
;         char right = p[2*ix + 1];
;         char max = max(left, right);
;         q[2*ix] = max;
;         q[2*ix+1] = 0 - max;
;     }
; }
;}

;DISABLED_MASKED_STRIDED-LABEL: @masked_strided2(
;DISABLED_MASKED_STRIDED: vector.body:
;DISABLED_MASKED_STRIDED-NEXT:  %index = phi i32 
;DISABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.store.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;DISABLED_MASKED_STRIDED:        %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;DISABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shl nuw nsw <8 x i32> %[[VECIND]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
;DISABLED_MASKED_STRIDED-NEXT:  %[[M:.+]] = extractelement <8 x i1> %[[VMASK]], i32 0
;DISABLED_MASKED_STRIDED-NEXT:  br i1 %[[M]], label %pred.load.if, label %pred.load.continue
;DISABLED_MASKED_STRIDED-NOT:   %interleaved.mask =
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.load.
;DISABLED_MASKED_STRIDED-NOT:   call void @llvm.masked.store.
;DISABLED_MASKED_STRIDED-NOT:   %{{.*}} = shufflevector <16 x i8> %{{.*}}, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>

;ENABLED_MASKED_STRIDED-LABEL: @masked_strided2(
;ENABLED_MASKED_STRIDED: vector.body:
;ENABLED_MASKED_STRIDED-NEXT:  %index = phi i32
;ENABLED_MASKED_STRIDED-NEXT:  %[[VECIND:.+]] = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;ENABLED_MASKED_STRIDED:       %[[VMASK:.+]] = icmp ugt <8 x i32> %[[VECIND]], %{{broadcast.splat*}}
;ENABLED_MASKED_STRIDED:       %interleaved.mask = shufflevector <8 x i1> %[[VMASK]], <8 x i1> undef, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
;ENABLED_MASKED_STRIDED-NEXT:  %[[WIDEMASKEDLOAD:.+]] = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask, <16 x i8> undef)
;ENABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
;ENABLED_MASKED_STRIDED-NEXT:  %{{.*}} = shufflevector <16 x i8> %[[WIDEMASKEDLOAD]], <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
;ENABLED_MASKED_STRIDED:       call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %{{.*}}, <16 x i8>* %{{.*}}, i32 1, <16 x i1> %interleaved.mask)

; Function Attrs: norecurse nounwind
define dso_local void @masked_strided2(i8* noalias nocapture readonly %p, i8* noalias nocapture %q, i8 zeroext %guard) local_unnamed_addr  {
entry:
  %conv = zext i8 %guard to i32
  br label %for.body

for.body:
  %ix.024 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i32 %ix.024, %conv
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %mul = shl nuw nsw i32 %ix.024, 1
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %add = or i32 %mul, 1
  %arrayidx4 = getelementptr inbounds i8, i8* %p, i32 %add
  %1 = load i8, i8* %arrayidx4, align 1
  %cmp.i = icmp slt i8 %0, %1
  %spec.select.i = select i1 %cmp.i, i8 %1, i8 %0
  %arrayidx6 = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 %spec.select.i, i8* %arrayidx6, align 1
  %sub = sub i8 0, %spec.select.i
  %arrayidx11 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 %sub, i8* %arrayidx11, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.024, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
