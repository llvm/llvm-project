// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that dyn_groupprivate size is correctly cast to i32 when a
// different integer type is used, matching the uint32_t DynCGroupMem
// field in __tgt_kernel_arguments.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @target_dyn_groupprivate_i64(%size : i64) {
    omp.target dyn_groupprivate(%size : i64) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @target_dyn_groupprivate_i16(%size : i16) {
    omp.target dyn_groupprivate(%size : i16) {
      omp.terminator
    }
    llvm.return
  }

  llvm.func @target_dyn_groupprivate_i32(%size : i32) {
    omp.target dyn_groupprivate(%size : i32) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @target_dyn_groupprivate_i64
// CHECK: %[[TRUNC:.*]] = trunc i64 %{{.*}} to i32
// CHECK: %[[GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %{{.*}}, i32 0, i32 12
// CHECK: store i32 %[[TRUNC]], ptr %[[GEP]], align 4

// CHECK-LABEL: define void @target_dyn_groupprivate_i16
// CHECK: %[[ZEXT:.*]] = zext i16 %{{.*}} to i32
// CHECK: %[[GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %{{.*}}, i32 0, i32 12
// CHECK: store i32 %[[ZEXT]], ptr %[[GEP2]], align 4

// CHECK-LABEL: define void @target_dyn_groupprivate_i32
// CHECK-NOT: trunc
// CHECK-NOT: zext
// CHECK: %[[GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %{{.*}}, i32 0, i32 12
// CHECK: store i32 %{{.*}}, ptr %[[GEP3]], align 4
