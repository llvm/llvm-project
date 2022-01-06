; RUN: opt -loop-load-elim -S < %s | \
; RUN:     FileCheck %s -check-prefix=ALL -check-prefix=ONE_STRIDE_SPEC \
; RUN:                  -check-prefix=TWO_STRIDE_SPEC

; RUN: opt -loop-load-elim -S -enable-mem-access-versioning=0 < %s | \
; RUN:     FileCheck %s -check-prefix=ALL -check-prefix=NO_ONE_STRIDE_SPEC \
; RUN:                  -check-prefix=NO_TWO_STRIDE_SPEC

; RUN: opt -loop-load-elim -S -loop-load-elimination-scev-check-threshold=1 < %s | \
; RUN:     FileCheck %s -check-prefix=ALL -check-prefix=ONE_STRIDE_SPEC \
; RUN:                  -check-prefix=NO_TWO_STRIDE_SPEC

; Forwarding in the presence of symbolic strides:
;
;   for (unsigned i = 0; i < 100; i++)
;     A[i + 1] = A[Stride * i] + B[i];

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; ALL-LABEL: @f(
define void @f(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i64 %N,
               i64 %stride) {

; ONE_STRIDE_SPEC: %ident.check = icmp ne i64 %stride, 1

entry:
; NO_ONE_STRIDE_SPEC-NOT: %load_initial = load i32, i32* %A
; ONE_STRIDE_SPEC: %load_initial = load i32, i32* %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; NO_ONE_STRIDE_SPEC-NOT: %store_forwarded = phi i32 [ %load_initial, {{.*}} ], [ %add, %for.body ]
; ONE_STRIDE_SPEC: %store_forwarded = phi i32 [ %load_initial, {{.*}} ], [ %add, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %mul = mul i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  %load = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %load_1 = load i32, i32* %arrayidx2, align 4
; NO_ONE_STRIDE_SPEC-NOT: %add = add i32 %load_1, %store_forwarded
; ONE_STRIDE_SPEC: %add = add i32 %load_1, %store_forwarded
  %add = add i32 %load_1, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds i32, i32* %A, i64 %indvars.iv.next
  store i32 %add, i32* %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Similar to @f(), but with a struct type.
; ALL-LABEL: @f_struct(
define void @f_struct({ i32, i8 } * noalias nocapture %A, { i32, i8 }* noalias nocapture readonly %B, i64 %N,
               i64 %stride) {

; ONE_STRIDE_SPEC: %ident.check = icmp ne i64 %stride, 1

entry:
; NO_ONE_STRIDE_SPEC-NOT: %load_initial = load { i32, i8 }, { i32, i8 }* %A
; ONE_STRIDE_SPEC: %load_initial = load { i32, i8 }, { i32, i8 }* %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; NO_ONE_STRIDE_SPEC-NOT: %store_forwarded = phi { i32, i8 } [ %load_initial, {{.*}} ], [ %ins, %for.body ]
; ONE_STRIDE_SPEC: %store_forwarded = phi { i32, i8 } [  %load_initial, {{.*}} ], [ %ins, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %mul = mul i64 %indvars.iv, %stride
  %arrayidx = getelementptr inbounds { i32, i8 }, { i32, i8 }* %A, i64 %mul
  %load = load { i32, i8 }, { i32, i8 }* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds { i32, i8 }, { i32, i8 }* %B, i64 %indvars.iv
  %load_1 = load { i32, i8 }, { i32, i8 }* %arrayidx2, align 4

; NO_ONE_STRIDE_SPEC-NOT: %v1 = extractvalue { i32, i8 } %store_forwarded
; ONE_STRIDE_SPEC: %v1 = extractvalue { i32, i8 } %store_forwarded
; ONE_STRIDE_SPEC: %add = add i32 %v1, %v2

  %v1 = extractvalue { i32, i8 } %load, 0
  %v2 = extractvalue { i32, i8} %load_1, 0
  %add = add i32 %v1, %v2
  %ins = insertvalue { i32, i8 } undef, i32 %add, 0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds { i32, i8 }, { i32, i8 }* %A, i64 %indvars.iv.next
  store { i32, i8 } %ins, { i32, i8 }* %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; With two symbolic strides:
;
;   for (unsigned i = 0; i < 100; i++)
;     A[Stride2 * (i + 1)] = A[Stride1 * i] + B[i];

; ALL-LABEL: @two_strides(
define void @two_strides(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i64 %N,
                         i64 %stride.1, i64 %stride.2) {

; TWO_STRIDE_SPEC: %ident.check = icmp ne i64 %stride.2, 1
; TWO_STRIDE_SPEC: %ident.check1 = icmp ne i64 %stride.1, 1
; NO_TWO_STRIDE_SPEC-NOT: %ident.check{{.*}} = icmp ne i64 %stride{{.*}}, 1

entry:
; NO_TWO_STRIDE_SPEC-NOT: %load_initial = load i32, i32* %A
; TWO_STRIDE_SPEC: %load_initial = load i32, i32* %A
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
; NO_TWO_STRIDE_SPEC-NOT: %store_forwarded = phi i32 [ %load_initial, {{.*}} ], [ %add, %for.body ]
; TWO_STRIDE_SPEC: %store_forwarded = phi i32 [ %load_initial, {{.*}} ], [ %add, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %mul = mul i64 %indvars.iv, %stride.1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %mul
  %load = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %load_1 = load i32, i32* %arrayidx2, align 4
; NO_TWO_STRIDE_SPEC-NOT: %add = add i32 %load_1, %store_forwarded
; TWO_STRIDE_SPEC: %add = add i32 %load_1, %store_forwarded
  %add = add i32 %load_1, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %mul.2 = mul i64 %indvars.iv.next, %stride.2
  %arrayidx_next = getelementptr inbounds i32, i32* %A, i64 %mul.2
  store i32 %add, i32* %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
