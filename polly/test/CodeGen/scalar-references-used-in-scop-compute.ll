; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; Test the code generation in the presence of a scalar out-of-scop value being
; used from within the SCoP.

; CHECH-LABEL: @scalar-function-argument
; CHECK: polly.split_new_and_old


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @scalar-function-argument(ptr %A, float %sqrinv) {
entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %mul104 = fmul float 1.0, %sqrinv
  %rp107 = getelementptr float, ptr %A, i64 %indvar
  store float %mul104, ptr %rp107, align 4
  %indvar.next = add nsw i64 %indvar, 1
  %cmp = icmp slt i64 1024, %indvar.next
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

; CHECH-LABEL: @scalar-outside-of-scop
; CHECK: polly.split_new_and_old

define void @scalar-outside-of-scop(ptr %A) {
entry:
  %sqrinv = call float @getFloat()
  br label %for.body

for.body:
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
  %mul104 = fmul float 1.0, %sqrinv
  %rp107 = getelementptr float, ptr %A, i64 %indvar
  store float %mul104, ptr %rp107, align 4
  %indvar.next = add nsw i64 %indvar, 1
  %cmp = icmp slt i64 1024, %indvar.next
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

declare float @getFloat()
