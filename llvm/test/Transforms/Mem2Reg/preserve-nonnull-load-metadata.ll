; RUN: opt < %s -passes=mem2reg -S | FileCheck %s

; This tests that mem2reg preserves the !nonnull metadata on loads
; from allocas that get optimized out.

; Check the case where the alloca in question has a single store.
define float* @single_store(float** %arg) {
; CHECK-LABEL: define float* @single_store
; CHECK: %arg.load = load float*, float** %arg, align 8
; CHECK: [[ASSUME:%(.*)]] = icmp ne float* %arg.load, null
; CHECK: call void @llvm.assume(i1 {{.*}}[[ASSUME]])
; CHECK: ret float* %arg.load
entry:
  %buf = alloca float*
  %arg.load = load float*, float** %arg, align 8
  store float* %arg.load, float** %buf, align 8
  %buf.load = load float*, float **%buf, !nonnull !0
  ret float* %buf.load
}

; Check the case where the alloca in question has more than one
; store but still within one basic block.
define float* @single_block(float** %arg) {
; CHECK-LABEL: define float* @single_block
; CHECK: %arg.load = load float*, float** %arg, align 8
; CHECK: [[ASSUME:%(.*)]] = icmp ne float* %arg.load, null
; CHECK: call void @llvm.assume(i1 {{.*}}[[ASSUME]])
; CHECK: ret float* %arg.load
entry:
  %buf = alloca float*
  %arg.load = load float*, float** %arg, align 8
  store float* null, float** %buf, align 8
  store float* %arg.load, float** %buf, align 8
  %buf.load = load float*, float **%buf, !nonnull !0
  ret float* %buf.load
}

; Check the case where the alloca in question has more than one
; store and also reads ands writes in multiple blocks.
define float* @multi_block(float** %arg) {
; CHECK-LABEL: define float* @multi_block
; CHECK-LABEL: entry:
; CHECK: %arg.load = load float*, float** %arg, align 8
; CHECK: br label %next
; CHECK-LABEL: next:
; CHECK: [[ASSUME:%(.*)]] = icmp ne float* %arg.load, null
; CHECK: call void @llvm.assume(i1 {{.*}}[[ASSUME]])
; CHECK: ret float* %arg.load
entry:
  %buf = alloca float*
  %arg.load = load float*, float** %arg, align 8
  store float* null, float** %buf, align 8
  br label %next
next:
  store float* %arg.load, float** %buf, align 8
  %buf.load = load float*, float** %buf, !nonnull !0
  ret float* %buf.load
}

; Check that we don't add an assume if it's not
; necessary i.e. the value is already implied to be nonnull
define float* @no_assume(float** %arg) {
; CHECK-LABEL: define float* @no_assume
; CHECK-LABEL: entry:
; CHECK: %arg.load = load float*, float** %arg, align 8
; CHECK: %cn = icmp ne float* %arg.load, null
; CHECK: br i1 %cn, label %next, label %fin
; CHECK-LABEL: next:
; CHECK-NOT: call void @llvm.assume
; CHECK: ret float* %arg.load
; CHECK-LABEL: fin:
; CHECK: ret float* null
entry:
  %buf = alloca float*
  %arg.load = load float*, float** %arg, align 8
  %cn = icmp ne float* %arg.load, null
  br i1 %cn, label %next, label %fin
next:
; At this point the above nonnull check ensures that
; the value %arg.load is nonnull in this block and thus
; we need not add the assume.
  store float* %arg.load, float** %buf, align 8
  %buf.load = load float*, float** %buf, !nonnull !0
  ret float* %buf.load
fin:
  ret float* null
}

!0 = !{}
