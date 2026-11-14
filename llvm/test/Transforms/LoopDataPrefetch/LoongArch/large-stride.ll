; RUN: opt --mtriple=loongarch64 -mattr=+d --passes=loop-data-prefetch -S < %s | FileCheck %s --check-prefixes=ALL,GENERIC
; RUN: opt --mtriple=loongarch64 -mcpu=la464 --passes=loop-data-prefetch -S < %s | FileCheck %s --check-prefixes=ALL,LA464
; RUN: opt --mtriple=loongarch64 -mcpu=la664 --passes=loop-data-prefetch -S < %s | FileCheck %s --check-prefixes=ALL,LA664

;; LA664 only gets software prefetches for strides of at least 2 KiB, which its
;; hardware prefetchers do not track. Untuned CPUs get none.

; ALL-LABEL: @stride_1024(
define void @stride_1024(ptr nocapture %a, ptr nocapture readonly %b) {
entry:
  br label %for.body

; ALL: for.body:
for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %iv
; ALL-NOT: call void @llvm.prefetch
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, ptr %a, i64 %iv
  store double %add, ptr %arrayidx2, align 8
  %iv.next = add nuw nsw i64 %iv, 128
  %exitcond = icmp eq i64 %iv.next, 128000
  br i1 %exitcond, label %for.end, label %for.body

; ALL: for.end:
for.end:
  ret void
}

; ALL-LABEL: @stride_2048(
define void @stride_2048(ptr nocapture %a, ptr nocapture readonly %b) {
entry:
  br label %for.body

; ALL: for.body:
for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %iv
; GENERIC-NOT: call void @llvm.prefetch
; LA464: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
; LA664: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, ptr %a, i64 %iv
; LA464: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 1, i32 3, i32 1)
; LA664: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 1, i32 3, i32 1)
  store double %add, ptr %arrayidx2, align 8
  %iv.next = add nuw nsw i64 %iv, 256
  %exitcond = icmp eq i64 %iv.next, 256000
  br i1 %exitcond, label %for.end, label %for.body

; ALL: for.end:
for.end:
  ret void
}
