// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks that the begin-pointer offset for a map of a single element
// of a multi-dimensional Fortran array (accessed through a descriptor, so the
// map has a non-array base type and carries per-dimension bounds) is linearized
// using the per-dimension strides carried on the omp.map.bounds operations,
// rather than the section extents. For a single-element section all extents are
// 1, so an extent-based fold would collapse every element onto the same begin
// pointer. The offset must instead be:
//
//   sum_d lower_bound[d] * stride[d]
//
// Here the strides are provided in bytes (stride_in_bytes(true)) with distinct
// values 8 and 256, so the offset is lb0*8 + lb1*256 applied through a byte
// (i8) GEP.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @single_element_2d_map(%arg0: !llvm.ptr, %lb0: i64, %lb1: i64) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %stride0 = llvm.mlir.constant(8 : i64) : i64
    %stride1 = llvm.mlir.constant(256 : i64) : i64
    %baddr = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %b0 = omp.map.bounds lower_bound(%lb0 : i64) upper_bound(%lb0 : i64) extent(%c1 : i64) stride(%stride0 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %b1 = omp.map.bounds lower_bound(%lb1 : i64) upper_bound(%lb1 : i64) extent(%c1 : i64) stride(%stride1 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %m0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%baddr : !llvm.ptr, f64) bounds(%b0, %b1) name("arr(i,j)") -> !llvm.ptr
    %m1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>) map_clauses(tofrom) capture(ByRef) members(%m0 : [0] : !llvm.ptr) name("arr(i,j)") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m0 -> %a0, %m1 -> %a1 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @single_element_2d_map(ptr %[[ARG0:.*]], i64 %[[LB0:.*]], i64 %[[LB1:.*]])
// CHECK: %[[T0:.*]] = mul i64 %[[LB0]], 8
// CHECK: %[[OFF0:.*]] = add i64 0, %[[T0]]
// CHECK: %[[T1:.*]] = mul i64 %[[LB1]], 256
// CHECK: %[[OFF:.*]] = add i64 %[[OFF0]], %[[T1]]
// CHECK: %[[BASE:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK: %[[ARR_OFFSET:.*]] = getelementptr inbounds i8, ptr %[[BASE]], i64 %[[OFF]]