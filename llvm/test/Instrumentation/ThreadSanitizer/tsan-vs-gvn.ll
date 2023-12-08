; RUN: opt < %s -aa-pipeline=basic-aa -passes='gvn,tsan' -S | FileCheck %s
; TSAN conflicts with load widening. Make sure the load widening is off with -tsan.

; 32-bit little endian target.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

%struct_of_8_bytes_4_aligned = type { i32, i8, i8, i8, i8}

@f = global %struct_of_8_bytes_4_aligned zeroinitializer, align 4

; Accessing bytes 4 and 6, not ok to widen to i32 if sanitize_thread is set.

define i32 @test_widening_bad(ptr %P) nounwind ssp noredzone sanitize_thread {
entry:
  %tmp = load i8, ptr getelementptr inbounds (%struct_of_8_bytes_4_aligned, ptr @f, i64 0, i32 1), align 4
  %conv = zext i8 %tmp to i32
  %tmp1 = load i8, ptr getelementptr inbounds (%struct_of_8_bytes_4_aligned, ptr @f, i64 0, i32 3), align 1
  %conv2 = zext i8 %tmp1 to i32
  %add = add nsw i32 %conv, %conv2
  ret i32 %add
; CHECK: @test_widening_bad
; CHECK: call void @__tsan_read1
; CHECK: call void @__tsan_read1
; CHECK-NOT: call void @__tsan_read4
; CHECK: ret i32
}
