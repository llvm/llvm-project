; RUN: opt -loop-load-elim -mtriple=aarch64 -mattr=+sve -S -debug < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Regression tests verifying "assumption that TypeSize is not scalable" and
; "Invalid size request on a scalable vector." are not produced by
; -load-loop-elim (this would cause the test to fail because opt would exit with
; a non-zero exit status).

; No output checked for this one, but causes a fatal error if the regression is present.

define void @regression_test_get_gep_induction_operand_typesize_warning(i64 %n, <vscale x 4 x i32>* %a) {
entry:
  br label %loop.body

loop.body:
  %0 = phi i64 [ 0, %entry ], [ %1, %loop.body ]
  %idx = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %a, i64 %0
  store <vscale x 4 x i32> zeroinitializer, <vscale x 4 x i32>* %idx
  %1 = add i64 %0, 1
  %2 = icmp eq i64 %1, %n
  br i1 %2, label %loop.end, label %loop.body

loop.end:
  ret void
}

; CHECK-LABEL: LAA: Found a loop in regression_test_loop_access_scalable_typesize
; CHECK: LAA: Bad stride - Scalable object:
define void @regression_test_loop_access_scalable_typesize(<vscale x 16 x i8>* %input_ptr) {
entry:
  br label %vector.body
vector.body:
  %ind_ptr = phi <vscale x 16 x i8>* [ %next_ptr, %vector.body ], [ %input_ptr, %entry ]
  %ind = phi i64 [ %next, %vector.body ], [ 0, %entry ]
  %ld = load <vscale x 16 x i8>, <vscale x 16 x i8>* %ind_ptr, align 16
  store <vscale x 16 x i8> zeroinitializer, <vscale x 16 x i8>* %ind_ptr, align 16
  %next_ptr = getelementptr inbounds <vscale x 16 x i8>, <vscale x 16 x i8>* %ind_ptr, i64 1
  %next = add i64 %ind, 1
  %cond = icmp ult i64 %next, 1024
  br i1 %cond, label %end, label %vector.body
end:
  ret void
}

; CHECK-LABEL: LAA: Found a loop in regression_test_loop_access_scalable_typesize_nonscalable_object
; CHECK: LAA: Bad stride - Scalable object:
define void @regression_test_loop_access_scalable_typesize_nonscalable_object(i8* %input_ptr) {
entry:
  br label %vector.body
vector.body:
  %ind_ptr = phi i8* [ %next_ptr, %vector.body ], [ %input_ptr, %entry ]
  %ind = phi i64 [ %next, %vector.body ], [ 0, %entry ]
  %scalable_ptr = bitcast i8* %ind_ptr to <vscale x 16 x i8>*
  %ld = load <vscale x 16 x i8>, <vscale x 16 x i8>* %scalable_ptr, align 16
  store <vscale x 16 x i8> zeroinitializer, <vscale x 16 x i8>* %scalable_ptr, align 16
  %next_ptr = getelementptr inbounds i8, i8* %ind_ptr, i64 1
  %next = add i64 %ind, 1
  %cond = icmp ult i64 %next, 1024
  br i1 %cond, label %end, label %vector.body
end:
  ret void
}
