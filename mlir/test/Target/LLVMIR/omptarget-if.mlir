// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @target_if_variable(%x : i1) {
    omp.target if(%x) {
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define void @target_if_variable(
  // CHECK-SAME: i1 %[[IF_COND:.*]])
  // CHECK: br i1 %[[IF_COND]], label %[[THEN_LABEL:.*]], label %[[ELSE_LABEL:.*]]

  // CHECK: [[THEN_LABEL]]:
  // CHECK-NOT: {{^.*}}:
  // CHECK: %[[RC:.*]] = call i32 @__tgt_target_kernel
  // CHECK-NEXT: %[[OFFLOAD_SUCCESS:.*]] = icmp ne i32 %[[RC]], 0
  // CHECK-NEXT: br i1 %[[OFFLOAD_SUCCESS]], label %[[OFFLOAD_FAIL_LABEL:.*]], label %[[OFFLOAD_CONT_LABEL:.*]]

  // CHECK: [[OFFLOAD_FAIL_LABEL]]:
  // CHECK-NEXT: call void @[[FALLBACK_FN:__omp_offloading_.*_.*_target_if_variable_l.*]]()
  // CHECK-NEXT: br label %[[OFFLOAD_CONT_LABEL]]

  // CHECK: [[OFFLOAD_CONT_LABEL]]:
  // CHECK-NEXT: br label %[[END_LABEL:.*]]

  // CHECK: [[ELSE_LABEL]]:
  // CHECK-NEXT: call void @[[FALLBACK_FN]]()
  // CHECK-NEXT: br label %[[END_LABEL]]

  llvm.func @target_if_true() {
    %0 = llvm.mlir.constant(true) : i1
    omp.target if(%0) {
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define void @target_if_true()
  // CHECK-NOT: {{^.*}}:
  // CHECK: br label %[[ENTRY:.*]]

  // CHECK: [[ENTRY]]:
  // CHECK-NOT: {{^.*}}:
  // CHECK: %[[RC:.*]] = call i32 @__tgt_target_kernel
  // CHECK-NEXT: %[[OFFLOAD_SUCCESS:.*]] = icmp ne i32 %[[RC]], 0
  // CHECK-NEXT: br i1 %[[OFFLOAD_SUCCESS]], label %[[OFFLOAD_FAIL_LABEL:.*]], label %[[OFFLOAD_CONT_LABEL:.*]]

  // CHECK: [[OFFLOAD_FAIL_LABEL]]:
  // CHECK-NEXT: call void @[[FALLBACK_FN:.*]]()
  // CHECK-NEXT: br label %[[OFFLOAD_CONT_LABEL]]

  llvm.func @target_if_false() {
    %0 = llvm.mlir.constant(false) : i1
    omp.target if(%0) {
      omp.terminator
    }
    llvm.return
  }

  // CHECK-LABEL: define void @target_if_false()
  // CHECK-NEXT: br label %[[ENTRY:.*]]

  // CHECK: [[ENTRY]]:
  // CHECK-NEXT: call void @__omp_offloading_{{.*}}_{{.*}}_target_if_false_l{{.*}}()
}

