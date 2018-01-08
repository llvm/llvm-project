; RUN: opt %loadPolly -S -polly-codegen-ppcg \
; RUN: -polly-use-llvm-names < %s
; ModuleID = 'test/GPGPU/zero-size-array.ll'

; REQUIRES: pollyacc

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"


; We used to divide the element size by 8 to arrive at the 'actual' size
; of an array element. This used to cause arrays that have an element size
; of less than 8 to collapse to size 0. This test makes sure that it does
; not happen anymore.

; f(int *niters_ptr, int *arr[0]) {
;     const int inters = *niters_ptr;
;     for(int i = 0; i < niters; i++) {
;       arr[0][i + 1] = 0
;     }
; }

; Function Attrs: nounwind uwtable
define void @f(i32* noalias %niters.ptr, [0 x i32]* noalias %arr) #0 {
entry:
  %niters = load i32, i32* %niters.ptr, align 4
  br label %loop.body

loop.body:                                             ; preds = %loop.body, %entry
  %indvar = phi i32 [ %indvar.next, %loop.body ], [ 1, %entry ]
  %indvar.sext = sext i32 %indvar to i64
  %arr.slot = getelementptr [0 x i32], [0 x i32]* %arr, i64 0, i64 %indvar.sext
  store i32 0, i32* %arr.slot, align 4
  %tmp8 = icmp eq i32 %indvar, %niters
  %indvar.next = add i32 %indvar, 1
  br i1 %tmp8, label %loop.exit, label %loop.body

loop.exit:                                    ; preds = %loop.body
  %tmp10 = icmp sgt i32 undef, 0
  br label %auxiliary.loop

auxiliary.loop:                                            ; preds = %"101", %loop.exit
  %tmp11 = phi i1 [ %tmp10, %loop.exit ], [ undef, %auxiliary.loop ]
  br i1 undef, label %auxiliary.loop, label %exit

exit:                              ; preds = %auxiliary.loop
  ret void
}

attributes #0 = { nounwind uwtable }
