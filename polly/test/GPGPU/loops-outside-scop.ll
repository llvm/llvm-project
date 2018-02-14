; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s -check-prefix=SCOP

; There is no FileCheck because we want to make sure that this doesn't crash.
; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-fail-on-verify-module-failure \
; RUN: -disable-output < %s

; REQUIRES: pollyacc

; Due to the existence of the `fence` call, We can only detect the inner loop
; and not the outer loop. PPCGCodeGeneration had not implemented this case.
; The fix was to pull the implementation from `IslNodeBuilder.

; Make sure that we only capture the inner loop
; SCOP:      Function: f
; SCOP-NEXT: Region: %for2.body---%for2.body.fence
; SCOP-NEXT: Max Loop Depth:  1

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @fn_to_fence(i32 *%val)

; void f(int *arr, bool shouldcont) {
;     for(int i = 0; ; i++) {
;         for(int j = 0; j < 10; j++) {
;             arr[j] = i;
;         }
;         fence(arr);
;         if (!shouldcont) break;
;     }
; }


; Function Attrs: nounwind uwtable
define void @f(i32 *%arr, i1 %shouldcont) #1 {
entry:
  br label %for.init

for.init:                                             ; preds = %for.end, %entry.split
  %i = phi i32 [ %i.next, %for.end ], [ 0, %entry ]
  br label %for2.body

for2.body:                                             ; preds = %"65", %"64"
  %j = phi i32 [ %j.next, %for2.body ], [ 0, %for.init ]
  %j.sext = sext i32 %j to i64
  %arr.slot = getelementptr i32, i32* %arr, i64 %j.sext
  store i32 %i, i32* %arr.slot, align 4
  %exitcond = icmp eq i32 %j, 10
  %j.next = add i32 %j, 1
  br i1 %exitcond, label %for2.body.fence, label %for2.body

for2.body.fence:                                             ; preds = %"65"
  call void @fn_to_fence(i32* %arr) #2
  br i1 %shouldcont, label %for.end, label %exit
for.end:                                             ; preds = %"69"
  %i.next = add i32 %i, 1
  br label %for.init

exit:                                             ; preds = %"69"
  ret void

}


attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }
