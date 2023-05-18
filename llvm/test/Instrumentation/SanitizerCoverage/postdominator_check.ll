; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc -sanitizer-coverage-prune-blocks=1 -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc -sanitizer-coverage-prune-blocks=0 -S | FileCheck %s --check-prefix=CHECK_NO_PRUNE

define i32 @foo(i32) #0 {
  %2 = icmp sgt i32 %0, 0
  br i1 %2, label %left, label %right
; CHECK: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

left:
  %3 = icmp sgt i32 %0, 10
  br i1 %3, label %left_left, label %left_right
; CHECK-LABEL: left:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: left:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

left_left:
  br label %left_join
; CHECK-LABEL: left_left:
; CHECK: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: left_left:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

left_right:
  br label %left_join
; CHECK-LABEL: left_right:
; CHECK: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: left_right:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

left_join:
  br label %finish
; CHECK-LABEL: left_join:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: left_join:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

right:
  %4 = icmp sgt i32 %0, 10
  br i1 %4, label %right_left, label %right_right
; CHECK-LABEL: right:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: right:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

right_left:
  br label %right_join
; CHECK-LABEL: right_left:
; CHECK: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: right_left:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

right_right:
  br label %right_join
; CHECK-LABEL: right_right:
; CHECK: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: right_right:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

right_join:
  br label %finish
; CHECK-LABEL: right_join:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: right_join:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

finish:
  ret i32 %0
; CHECK-LABEL: finish:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()

; CHECK_NO_PRUNE-LABEL: finish:
; CHECK_NO_PRUNE: call void @__sanitizer_cov_trace_pc()

}
