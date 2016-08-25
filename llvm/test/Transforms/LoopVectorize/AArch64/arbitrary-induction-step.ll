; RUN: opt -S < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=4 | FileCheck %s
; RUN: opt -S < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 | FileCheck %s --check-prefix=FORCE-VEC

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnueabi"

; Test integer induction variable of step 2:
;   for (int i = 0; i < 1024; i+=2) {
;     int tmp = *A++;
;     sum += i * tmp;
;   }

; CHECK-LABEL: @ind_plus2(
; CHECK: load <4 x i32>, <4 x i32>*
; CHECK: load <4 x i32>, <4 x i32>*
; CHECK: mul nsw <4 x i32>
; CHECK: mul nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: %index.next = add i64 %index, 8
; CHECK: icmp eq i64 %index.next, 512

; FORCE-VEC-LABEL: @ind_plus2(
; FORCE-VEC: %wide.load = load <2 x i32>, <2 x i32>*
; FORCE-VEC: mul nsw <2 x i32>
; FORCE-VEC: add nsw <2 x i32>
; FORCE-VEC: %index.next = add i64 %index, 2
; FORCE-VEC: icmp eq i64 %index.next, 512
define i32 @ind_plus2(i32* %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %A.addr = phi i32* [ %A, %entry ], [ %inc.ptr, %for.body ]
  %i = phi i32 [ 0, %entry ], [ %add1, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %inc.ptr = getelementptr inbounds i32, i32* %A.addr, i64 1
  %0 = load i32, i32* %A.addr, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %add1 = add nsw i32 %i, 2
  %cmp = icmp slt i32 %add1, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa
}


; Test integer induction variable of step -2:
;   for (int i = 1024; i > 0; i-=2) {
;     int tmp = *A++;
;     sum += i * tmp;
;   }

; CHECK-LABEL: @ind_minus2(
; CHECK: load <4 x i32>, <4 x i32>*
; CHECK: load <4 x i32>, <4 x i32>*
; CHECK: mul nsw <4 x i32>
; CHECK: mul nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: %index.next = add i64 %index, 8
; CHECK: icmp eq i64 %index.next, 512

; FORCE-VEC-LABEL: @ind_minus2(
; FORCE-VEC: %wide.load = load <2 x i32>, <2 x i32>*
; FORCE-VEC: mul nsw <2 x i32>
; FORCE-VEC: add nsw <2 x i32>
; FORCE-VEC: %index.next = add i64 %index, 2
; FORCE-VEC: icmp eq i64 %index.next, 512
define i32 @ind_minus2(i32* %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %A.addr = phi i32* [ %A, %entry ], [ %inc.ptr, %for.body ]
  %i = phi i32 [ 1024, %entry ], [ %sub, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %inc.ptr = getelementptr inbounds i32, i32* %A.addr, i64 1
  %0 = load i32, i32* %A.addr, align 4
  %mul = mul nsw i32 %0, %i
  %add = add nsw i32 %mul, %sum
  %sub = add nsw i32 %i, -2
  %cmp = icmp sgt i32 %i, 2
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa
}


; Test pointer induction variable of step 2. As currently we don't support
; masked load/store, vectorization is possible but not beneficial. If loop
; vectorization is not enforced, LV will only do interleave.
;   for (int i = 0; i < 1024; i++) {
;     int tmp0 = *A++;
;     int tmp1 = *A++;
;     sum += tmp0 * tmp1;
;   }

; CHECK-LABEL: @ptr_ind_plus2(
; CHECK: %[[V0:.*]] = load <8 x i32>
; CHECK: %[[V1:.*]] = load <8 x i32>
; CHECK: shufflevector <8 x i32> %[[V0]], <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <8 x i32> %[[V1]], <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <8 x i32> %[[V0]], <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: shufflevector <8 x i32> %[[V1]], <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK: mul nsw <4 x i32>
; CHECK: mul nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: %index.next = add i64 %index, 8
; CHECK: icmp eq i64 %index.next, 1024

; FORCE-VEC-LABEL: @ptr_ind_plus2(
; FORCE-VEC: %[[V:.*]] = load <4 x i32>
; FORCE-VEC: shufflevector <4 x i32> %[[V]], <4 x i32> undef, <2 x i32> <i32 0, i32 2>
; FORCE-VEC: shufflevector <4 x i32> %[[V]], <4 x i32> undef, <2 x i32> <i32 1, i32 3>
; FORCE-VEC: mul nsw <2 x i32>
; FORCE-VEC: add nsw <2 x i32>
; FORCE-VEC: %index.next = add i64 %index, 2
; FORCE-VEC: icmp eq i64 %index.next, 1024
define i32 @ptr_ind_plus2(i32* %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %A.addr = phi i32* [ %A, %entry ], [ %inc.ptr1, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %inc.ptr = getelementptr inbounds i32, i32* %A.addr, i64 1
  %0 = load i32, i32* %A.addr, align 4
  %inc.ptr1 = getelementptr inbounds i32, i32* %A.addr, i64 2
  %1 = load i32, i32* %inc.ptr, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %sum
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa
}
