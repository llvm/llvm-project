// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Descriptor strides are bytes and need not describe contiguous storage, even
// for the first dimension. Preserve all four source strides and add a final
// byte dimension so that each selected element transfers exactly four bytes.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @descriptor_update(%descriptor: !llvm.ptr, %stride0: i64, %stride1: i64, %stride2: i64, %stride3: i64) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c5 = llvm.mlir.constant(5 : i64) : i64
    %b0 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c2 : i64) extent(%c3 : i64) stride(%stride0 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %b1 = omp.map.bounds lower_bound(%c2 : i64) upper_bound(%c4 : i64) extent(%c5 : i64) stride(%stride1 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %b2 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) extent(%c2 : i64) stride(%stride2 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %b3 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) extent(%c2 : i64) stride(%stride3 : i64) start_idx(%c1 : i64) stride_in_bytes(true)
    %to = omp.map.info var_ptr(%descriptor : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<4 x array<3 x i64>>)>) map_clauses(to) capture(ByRef) var_ptr_ptr(%descriptor : !llvm.ptr, i32) bounds(%b0, %b1, %b2, %b3) -> !llvm.ptr
    omp.target_update map_entries(%to : !llvm.ptr)
    %from = omp.map.info var_ptr(%descriptor : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<4 x array<3 x i64>>)>) map_clauses(from) capture(ByRef) var_ptr_ptr(%descriptor : !llvm.ptr, i32) bounds(%b0, %b1, %b2, %b3) -> !llvm.ptr
    omp.target_update map_entries(%from : !llvm.ptr)
    llvm.return
  }
}

// CHECK: @.offload_sizes = {{.*}}[1 x i64] [i64 5]
// CHECK: @.offload_maptypes = {{.*}}[1 x i64] [i64 17592186044417]
// CHECK: @.offload_sizes.1 = {{.*}}[1 x i64] [i64 5]
// CHECK: @.offload_maptypes.2 = {{.*}}[1 x i64] [i64 17592186044418]
// CHECK-LABEL: define void @descriptor_update(
// CHECK-SAME: ptr %[[DESC:.*]], i64 %[[S0:.*]], i64 %[[S1:.*]], i64 %[[S2:.*]], i64 %[[S3:.*]])
// CHECK: %[[FROM_DIMS:.*]] = alloca [5 x %struct.descriptor_dim.0]
// CHECK: %[[TO_DIMS:.*]] = alloca [5 x %struct.descriptor_dim]
// CHECK: %[[BAND_OFFSET:.*]] = mul i64 2, %[[S1]]
// CHECK: %[[BASE:.*]] = load ptr, ptr %[[DESC]]
// CHECK: %[[IS_NULL:.*]] = icmp eq ptr %[[BASE]], null
// CHECK: %[[INNER_COUNT:.*]] = select i1 %[[IS_NULL]], i64 0, i64 3
// CHECK: %[[BASE_ADDR:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: store ptr %[[BASE]], ptr %[[BASE_ADDR]]
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[TO_DIMS]], i64 0, i64 0
// CHECK: store i64 2,
// CHECK: store i64 %[[S3]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[TO_DIMS]], i64 0, i64 1
// CHECK: store i64 2,
// CHECK: store i64 %[[S2]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[TO_DIMS]], i64 0, i64 2
// CHECK: store i64 %[[BAND_OFFSET]],
// CHECK: store i64 3,
// CHECK: store i64 %[[S1]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[TO_DIMS]], i64 0, i64 3
// CHECK: store i64 %[[INNER_COUNT]],
// CHECK: store i64 %[[S0]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[TO_DIMS]], i64 0, i64 4
// CHECK: store i64 0,
// CHECK: store i64 4,
// CHECK: store i64 1,
// CHECK: store ptr %[[TO_DIMS]],
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_sizes, ptr @.offload_maptypes,
// CHECK: %[[FROM_BAND_OFFSET:.*]] = mul i64 2, %[[S1]]
// CHECK: %[[FROM_BASE:.*]] = load ptr, ptr %[[DESC]]
// CHECK: store ptr %[[FROM_BASE]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim.0], ptr %[[FROM_DIMS]], i64 0, i64 2
// CHECK: store i64 %[[FROM_BAND_OFFSET]],
// CHECK: store i64 3,
// CHECK: store i64 %[[S1]],
// CHECK: store ptr %[[FROM_DIMS]],
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_sizes.1, ptr @.offload_maptypes.2,

// -----

// Static array types provide complete source extents even when the bounds
// carry legacy unit strides. Reconstruct slice strides from those extents,
// rather than the selected section lengths. The scalars before and after the
// array retain their byte sizes and acquire no non-contiguous descriptors.
// A null raw array pointer represents an absent optional array and must give
// a zero first-dimension count, even though the section bounds are nonempty.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @explicit_array_update(%before: !llvm.ptr, %array: !llvm.ptr, %after: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %b0 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c2 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %b1 = omp.map.bounds lower_bound(%c2 : i64) upper_bound(%c4 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %b2 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %b3 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %first = omp.map.info var_ptr(%before : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr
    %section = omp.map.info var_ptr(%array : !llvm.ptr, !llvm.array<2 x array<2 x array<5 x array<3 x i32>>>>) map_clauses(from) capture(ByRef) bounds(%b0, %b1, %b2, %b3) -> !llvm.ptr
    %last = omp.map.info var_ptr(%after : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr
    omp.target_update map_entries(%first, %section, %last : !llvm.ptr, !llvm.ptr, !llvm.ptr)
    llvm.return
  }
}

// CHECK: @.offload_sizes = {{.*}}[3 x i64] [i64 4, i64 5, i64 4]
// CHECK: @.offload_maptypes = {{.*}}[3 x i64] [i64 2, i64 17592186044418, i64 2]
// CHECK-LABEL: define void @explicit_array_update(
// CHECK-SAME: ptr %[[BEFORE:.*]], ptr %[[ARRAY:.*]], ptr %[[AFTER:.*]])
// CHECK: %[[DIMS:.*]] = alloca [5 x %struct.descriptor_dim]
// CHECK-NOT: alloca {{.*}}%struct.descriptor_dim
// CHECK: %[[IS_NULL:.*]] = icmp eq ptr %[[ARRAY]], null
// CHECK-NEXT: %[[INNER_COUNT:.*]] = select i1 %[[IS_NULL]], i64 0, i64 3
// CHECK: store ptr %[[BEFORE]],
// CHECK: store ptr %[[ARRAY]],
// CHECK: store ptr %[[AFTER]],
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 0
// CHECK: store i64 0,
// CHECK: store i64 2,
// CHECK: store i64 120,
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 1
// CHECK: store i64 0,
// CHECK: store i64 2,
// CHECK: store i64 60,
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 2
// CHECK: store i64 24,
// CHECK: store i64 3,
// CHECK: store i64 12,
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 3
// CHECK: store i64 0,
// CHECK: store i64 %[[INNER_COUNT]],
// CHECK: store i64 4,
// CHECK: getelementptr inbounds [5 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 4
// CHECK: store i64 0,
// CHECK: store i64 4,
// CHECK: store i64 1,
// CHECK: %[[SECTION_PTR:.*]] = getelementptr inbounds [3 x ptr], ptr %{{.*}}, i32 0, i32 1
// CHECK-NEXT: store ptr %[[DIMS]], ptr %[[SECTION_PTR]]
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}i64 -1, i32 3, {{.*}}ptr @.offload_sizes, ptr @.offload_maptypes,

// -----

// A scalar subscript has extent one, but still retains the physical stride of
// the original array. The next stride includes the full source extent.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @scalar_subscript_update(%array: !llvm.ptr, %extent0: i64, %extent1: i64) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %b0 = omp.map.bounds lower_bound(%c1 : i64) upper_bound(%c1 : i64) extent(%c1 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %b1 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c1 : i64) extent(%extent1 : i64) stride(%extent0 : i64) start_idx(%c1 : i64)
    %section = omp.map.info var_ptr(%array : !llvm.ptr, i32) map_clauses(to) capture(ByRef) bounds(%b0, %b1) -> !llvm.ptr
    omp.target_update map_entries(%section : !llvm.ptr)
    llvm.return
  }
}

// CHECK: @.offload_sizes = {{.*}}[1 x i64] [i64 3]
// CHECK: @.offload_maptypes = {{.*}}[1 x i64] [i64 17592186044417]
// CHECK-LABEL: define void @scalar_subscript_update(
// CHECK-SAME: ptr %[[ARRAY:.*]], i64 %[[EXTENT:.*]], i64 %{{.*}})
// CHECK: %[[DIMS:.*]] = alloca [3 x %struct.descriptor_dim]
// CHECK: %[[OUTER_STRIDE:.*]] = mul i64 %[[EXTENT]], 4
// CHECK: %[[IS_NULL:.*]] = icmp eq ptr %[[ARRAY]], null
// CHECK-NEXT: %[[INNER_COUNT:.*]] = select i1 %[[IS_NULL]], i64 0, i64 1
// CHECK: getelementptr inbounds [3 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 0
// CHECK: store i64 2,
// CHECK: store i64 %[[OUTER_STRIDE]],
// CHECK: getelementptr inbounds [3 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 1
// CHECK: store i64 4,
// CHECK: store i64 %[[INNER_COUNT]],
// CHECK: store i64 4,
// CHECK: getelementptr inbounds [3 x %struct.descriptor_dim], ptr %[[DIMS]], i64 0, i64 2
// CHECK: store i64 0,
// CHECK: store i64 4,
// CHECK: store i64 1,
// CHECK: call void @__tgt_target_data_update_mapper

// -----

// A full innermost dimension followed by a section of the outermost dimension
// is contiguous. Keep the usual byte size and the selected starting address.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @contiguous_update(%array: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %b0 = omp.map.bounds lower_bound(%c0 : i64) upper_bound(%c2 : i64) stride(%c1 : i64) start_idx(%c1 : i64)
    %b1 = omp.map.bounds lower_bound(%c1 : i64) upper_bound(%c2 : i64) stride(%c3 : i64) start_idx(%c1 : i64)
    %section = omp.map.info var_ptr(%array : !llvm.ptr, !llvm.array<5 x array<3 x i32>>) map_clauses(to) capture(ByRef) bounds(%b0, %b1) -> !llvm.ptr
    omp.target_update map_entries(%section : !llvm.ptr)
    llvm.return
  }
}

// CHECK: @.offload_sizes = {{.*}}[1 x i64] [i64 24]
// CHECK: @.offload_maptypes = {{.*}}[1 x i64] [i64 1]
// CHECK-LABEL: define void @contiguous_update(
// CHECK-SAME: ptr %[[ARRAY:.*]])
// CHECK-NOT: %struct.descriptor_dim
// CHECK: %[[START:.*]] = getelementptr inbounds [5 x [3 x i32]], ptr %[[ARRAY]], i64 0, i64 1, i64 0
// CHECK: store ptr %[[START]],
// CHECK: call void @__tgt_target_data_update_mapper({{.*}}ptr @.offload_sizes, ptr @.offload_maptypes,
// CHECK: ret void
