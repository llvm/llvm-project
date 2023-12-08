; RUN: llc < %s | FileCheck %s

; MachineLICM should be able to hoist the sF reference out of the loop.

; CHECK: pushl %esi
; CHECK: pushl
; CHECK: movl  $176, %esi
; CHECK: addl  L___sF$non_lazy_ptr, %esi
; CHECK: .p2align  4, 0x90

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin8"

%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { ptr, i32 }
%struct.gcov_ctr_summary = type { i32, i32, i64, i64, i64 }
%struct.gcov_summary = type { i32, [1 x %struct.gcov_ctr_summary] }

@__sF = external global [0 x %struct.FILE]        ; <ptr> [#uses=1]

declare i32 @fprintf(ptr nocapture) nounwind

define void @gcov_exit() nounwind {
entry:
  br label %bb151

bb151:                                            ; preds = %bb59, %bb56, %bb14
  br i1 undef, label %bb56, label %bb59

bb56:                                             ; preds = %bb151
  %t0 = call i32 (ptr) @fprintf(ptr getelementptr inbounds ([0 x %struct.FILE], ptr @__sF, i32 0, i32 2)) nounwind
  br label %bb151

bb59:                                             ; preds = %bb151
  %t1 = call i32 (ptr) @fprintf(ptr getelementptr inbounds ([0 x %struct.FILE], ptr @__sF, i32 0, i32 2)) nounwind
  br label %bb151
}

