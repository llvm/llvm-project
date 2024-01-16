// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

!EntryArray = !llvm.struct<(!llvm.ptr, !llvm.ptr, i64, i32, i32)>
// CHECK: @__begin_offload_omp = internal constant [2 x %{{.*}}] [%{{.*}} { ptr @[[TGT_OP_1:.*]], ptr @[[TGT_OP_1_NAME:.*]], i64 0, i32 0, i32 0 }, %{{.*}} { ptr @[[TGT_OP_2:.*]], ptr @[[TGT_OP_2_NAME:.*]], i64 0, i32 0, i32 0 }]
// CHECK: @__end_offload_omp = constant ptr getelementptr inbounds (%{{.*}}, ptr @__begin_offload_omp, i64 2)
// CHECK: @[[TGT_OP_1]] = weak constant i8 0
// CHECK: @[[TGT_OP_1_NAME]] = internal unnamed_addr constant [{{.*}} x i8] c"{{.*}}0_0_main_l0\00"
// CHECK: @[[TGT_OP_2]] = weak constant i8 0
// CHECK: @[[TGT_OP_2_NAME]] = internal unnamed_addr constant [{{.*}} x i8] c"{{.*}}0_0_main_l1\00"
// CHECK: define void @main() {
// CHECK: %{{.*}} = call i32 @__tgt_target_kernel(ptr @{{.*}}, i64 -1, i32 -1, i32 0, ptr @[[TGT_OP_1]], ptr %{{.*}})
// CHECK: %{{.*}} = call i32 @__tgt_target_kernel(ptr @{{.*}}, i64 -1, i32 -1, i32 0, ptr @[[TGT_OP_2]], ptr %{{.*}})
// CHECK: }
// CHECK-LABEL: define internal void @{{.*}}0_0_main_l0() {
// CHECK-LABEL: define internal void @{{.*}}0_0_main_l1() {
module attributes {omp.is_target_device = false, omp.is_gpu = false} {
  llvm.mlir.global constant @__begin_offload_omp() : !llvm.array<0 x !EntryArray> {
    %zero = llvm.mlir.zero : !llvm.array<0 x !EntryArray>
    llvm.return %zero : !llvm.array<0 x !EntryArray>
  }
  llvm.mlir.global constant @__end_offload_omp() : !llvm.ptr {
    %array = llvm.mlir.addressof @__begin_offload_omp : !llvm.ptr
    llvm.return %array : !llvm.ptr
  }
  llvm.func @main() {
    omp.target info = #omp.tgt_entry_info<deviceID = 0, fileID = 0, line = 0, section = @omp> {
      omp.terminator
    }
    omp.target info = #omp.tgt_entry_info<deviceID = 0, fileID = 0, line = 1, section = @omp> {
      omp.terminator
    }
    llvm.return
  }
}

// -----

// CHECK: @[[TGT_OP:.*]] = weak constant i8 0
// CHECK: @[[TGT_OP_NAME:.*]] = internal unnamed_addr constant [{{.*}} x i8] c"{{.*}}0_0_main_l0\00"
// CHECK: @{{.*}} = weak constant %{{.*}} { ptr @[[TGT_OP]], ptr @[[TGT_OP_NAME]], i64 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
module attributes {omp.is_target_device = false, omp.is_gpu = false} {
  llvm.func @main() {
    omp.target info = #omp.tgt_entry_info<deviceID = 0, fileID = 0, line = 0> {
      omp.terminator
    }
    llvm.return
  }
}
