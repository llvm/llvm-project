// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// A private attach map describes the pointer or descriptor storage, whose size
// does not depend on whether its pointee is null. In particular, an absent
// optional Fortran array still needs a complete private descriptor on the
// device. The ordinary pointee and attach maps retain their null checks.

module attributes {omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp.clause_requires<none>, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 52>} {
  llvm.func @map_optional_descriptor(%descriptor: !llvm.ptr, %base_addr: !llvm.ptr) {
    %member = omp.map.info var_ptr(%descriptor : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<3 x i64>)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%base_addr : !llvm.ptr, f64) name("") -> !llvm.ptr
    %parent = omp.map.info var_ptr(%descriptor : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<3 x i64>)>) map_clauses(target_param, private, attach) capture(ByRef) var_ptr_ptr(%base_addr : !llvm.ptr, f64) members(%member : [0] : !llvm.ptr) name("optional_array") -> !llvm.ptr
    %attach = omp.map.info var_ptr(%descriptor : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<3 x i64>)>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%base_addr : !llvm.ptr, f64) name("optional_array") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%parent -> %arg0, %attach -> %arg1, %member -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: @.offload_sizes = private unnamed_addr constant [4 x i64] [i64 48, i64 0, i64 0, i64 0]
// CHECK: @.offload_maptypes = private unnamed_addr constant [4 x i64] [i64 16544, i64 3, i64 16384, i64 288]
// CHECK-LABEL: define void @map_optional_descriptor(
// CHECK-SAME: ptr %[[DESCRIPTOR:.*]], ptr %[[BASE_ADDR:.*]])
// CHECK: %[[PRIVATE_POINTEE:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
// CHECK: %[[ATTACH_POINTEE:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
// CHECK: %[[POINTEE:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
// CHECK-NOT: select {{.*}} i64 48
// CHECK: %[[IS_NULL:.*]] = icmp eq ptr %[[POINTEE]], null
// CHECK: %[[POINTEE_SIZE:.*]] = select i1 %[[IS_NULL]], i64 0, i64 8
// CHECK: %[[ATTACH_IS_NULL:.*]] = icmp eq ptr %[[ATTACH_POINTEE]], null
// CHECK-NEXT: %[[ATTACH_SIZE:.*]] = select i1 %[[ATTACH_IS_NULL]], i64 0, i64 48
// CHECK-NOT: getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
// CHECK: %[[POINTEE_SIZE_ADDR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 1
// CHECK-NEXT: store i64 %[[POINTEE_SIZE]], ptr %[[POINTEE_SIZE_ADDR]], align 8
// CHECK: %[[ATTACH_SIZE_ADDR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 2
// CHECK-NEXT: store i64 %[[ATTACH_SIZE]], ptr %[[ATTACH_SIZE_ADDR]], align 8
