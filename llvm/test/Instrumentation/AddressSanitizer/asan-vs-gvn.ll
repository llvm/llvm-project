; RUN: opt < %s -basic-aa -gvn -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s "-passes=function(require<basic-aa>,gvn),asan-pipeline" -S | FileCheck %s
; ASAN conflicts with load widening iff the widened load accesses data out of bounds
; (while the original unwidened loads do not).
; https://github.com/google/sanitizers/issues/20#issuecomment-136381262


; 32-bit little endian target.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

%struct_of_7_bytes_4_aligned = type { i32, i8, i8, i8}

@f = external global %struct_of_7_bytes_4_aligned , align 4

; Accessing bytes 4 and 6, not ok to widen to i32 if sanitize_address is set.

define i32 @test_widening_bad(i8* %P) nounwind ssp noredzone sanitize_address {
entry:
  %tmp = load i8, i8* getelementptr inbounds (%struct_of_7_bytes_4_aligned, %struct_of_7_bytes_4_aligned* @f, i64 0, i32 1), align 4
  %conv = zext i8 %tmp to i32
  %tmp1 = load i8, i8* getelementptr inbounds (%struct_of_7_bytes_4_aligned, %struct_of_7_bytes_4_aligned* @f, i64 0, i32 3), align 1
  %conv2 = zext i8 %tmp1 to i32
  %add = add nsw i32 %conv, %conv2
  ret i32 %add
; CHECK: @test_widening_bad
; CHECK: __asan_report_load1
; CHECK: __asan_report_load1
; CHECK-NOT: __asan_report
; We can not use check for "ret" here because __asan_report_load1 calls live after ret.
; CHECK: end_test_widening_bad
}

define void @end_test_widening_bad() {
  entry:
  ret void
}

;; Accessing bytes 4 and 5. No widen to i16.

define i32 @test_widening_ok(i8* %P) nounwind ssp noredzone sanitize_address {
entry:
  %tmp = load i8, i8* getelementptr inbounds (%struct_of_7_bytes_4_aligned, %struct_of_7_bytes_4_aligned* @f, i64 0, i32 1), align 4
  %conv = zext i8 %tmp to i32
  %tmp1 = load i8, i8* getelementptr inbounds (%struct_of_7_bytes_4_aligned, %struct_of_7_bytes_4_aligned* @f, i64 0, i32 2), align 1
  %conv2 = zext i8 %tmp1 to i32
  %add = add nsw i32 %conv, %conv2
  ret i32 %add
; CHECK: @test_widening_ok
; CHECK: __asan_report_load1
; CHECK: __asan_report_load1
; CHECK-NOT: __asan_report
; CHECK: end_test_widening_ok
}

define void @end_test_widening_ok() {
  entry:
  ret void
}
