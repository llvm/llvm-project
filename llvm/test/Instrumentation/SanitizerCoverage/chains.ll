; RUN: opt < %s -sancov -sanitizer-coverage-level=4 -sanitizer-coverage-trace-pc -sanitizer-coverage-prune-blocks=1  -S | FileCheck %s

define i32 @blah(i32) #0 {
  %2 = icmp sgt i32 %0, 1
  br i1 %2, label %branch, label %exit
; CHECK: call void @__sanitizer_cov_trace_pc()
branch:
  br label %pos2
; CHECK-LABEL: branch:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()
pos2:
  br label %pos3
; CHECK-LABEL: pos2:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()
pos3:
  br label %pos4
; CHECK-LABEL: pos3:
; CHECK-NOT: call void @__sanitizer_cov_trace_pc()
pos4:
  ret i32 0
; CHECK-LABEL: pos4:
; CHECK: call void @__sanitizer_cov_trace_pc()
exit:
  ret i32 0
; CHECK-LABEL: exit:
; CHECK: call void @__sanitizer_cov_trace_pc()
}
