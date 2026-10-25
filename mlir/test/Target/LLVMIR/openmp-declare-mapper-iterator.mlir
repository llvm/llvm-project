// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// A declare_mapper map clause with an iterator modifier yields one
// bounds-described omp.map.info per iteration: the array base stays in
// var_ptr and bounds select the mapped element, the same shape used for an
// ordinary array-section map. Each iteration is pushed at runtime via
// __tgt_push_mapper_component since the trip count is not statically known.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper_with_iterator : !llvm.struct<"mapper_type", (array<10 x i32>)> {
  ^bb0(%arg: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c9 = llvm.mlir.constant(9 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c9 step %c1) {
      %field = llvm.getelementptr %arg[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"mapper_type", (array<10 x i32>)>
      %bounds = omp.map.bounds lower_bound(%iv : i64) upper_bound(%iv : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %m = omp.map.info var_ptr(%field : !llvm.ptr, !llvm.array<10 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%bounds) name("") -> !llvm.ptr
      omp.yield(%m : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    omp.declare_mapper.info map_iterated(%it : !omp.iterated<!llvm.ptr>)
  }

  llvm.func @target_data_mapper_iterator(%addr : !llvm.ptr) {
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"mapper_type", (array<10 x i32>)>) map_clauses(tofrom) capture(ByRef) mapper(@mapper_with_iterator) name("") -> !llvm.ptr
    omp.target_data map_entries(%map : !llvm.ptr) {}
    llvm.return
  }
}

// CHECK-LABEL: define void @target_data_mapper_iterator
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 1, ptr %{{.*}}, ptr %{{.*}}, ptr @.offload_sizes, ptr @.offload_maptypes, ptr @.offload_mapnames, ptr %.offload_mappers)

// CHECK-LABEL: define internal void @.omp_mapper.mapper_with_iterator
// CHECK: omp_mapper.iterator.cond:
// CHECK: %[[CMP:.*]] = icmp ult i64 %{{.*}}, 10
// CHECK: br i1 %[[CMP]], label %omp_mapper.iterator.body, label %omp_mapper.iterator.exit
//
// The field GEP (struct index 0) is computed once per iteration, then the
// bounds' lower_bound selects the element within it via an array GEP.
// CHECK: omp_mapper.iterator.body:
// CHECK: %[[FIELD:.*]] = getelementptr %mapper_type, ptr %omp.arraymap.ptrcurrent, i32 0, i32 0
// CHECK: %[[ELEM:.*]] = getelementptr inbounds [10 x i32], ptr %[[FIELD]], i64 0, i64 %{{.*}}
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[ELEM]], ptr %[[ELEM]], i64 4, i64 %{{.*}}, ptr null)

// -----

// The iterated locator's mapper (its own `mapper(...)` on the yielded
// omp.map.info) must still be invoked for each element, the same as a
// statically-known member with a mapper.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @inner_mapper : !llvm.struct<"inner", (i32)> {
  ^bb0(%arg: !llvm.ptr):
    %m = omp.map.info var_ptr(%arg : !llvm.ptr, !llvm.struct<"inner", (i32)>) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.declare_mapper.info map_entries(%m : !llvm.ptr)
  }

  omp.declare_mapper @outer_mapper_with_iterator : !llvm.struct<"outer", (array<2 x struct<"inner", (i32)>>)> {
  ^bb0(%arg: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c1 step %c1) {
      %field = llvm.getelementptr %arg[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"outer", (array<2 x struct<"inner", (i32)>>)>
      %bounds = omp.map.bounds lower_bound(%iv : i64) upper_bound(%iv : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %m = omp.map.info var_ptr(%field : !llvm.ptr, !llvm.array<2 x struct<"inner", (i32)>>) map_clauses(tofrom) capture(ByRef) mapper(@inner_mapper) bounds(%bounds) name("") -> !llvm.ptr
      omp.yield(%m : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    omp.declare_mapper.info map_iterated(%it : !omp.iterated<!llvm.ptr>)
  }

  llvm.func @target_data_outer_mapper_iterator(%addr : !llvm.ptr) {
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"outer", (array<2 x struct<"inner", (i32)>>)>) map_clauses(tofrom) capture(ByRef) mapper(@outer_mapper_with_iterator) name("") -> !llvm.ptr
    omp.target_data map_entries(%map : !llvm.ptr) {}
    llvm.return
  }
}

// CHECK-LABEL: define internal void @.omp_mapper.outer_mapper_with_iterator
// CHECK: omp_mapper.iterator.body:
// CHECK: %[[FIELD2:.*]] = getelementptr %outer, ptr %omp.arraymap.ptrcurrent, i32 0, i32 0
// CHECK: %[[ELEM2:.*]] = getelementptr inbounds [2 x %inner], ptr %[[FIELD2]], i64 0, i64 %{{.*}}
// CHECK: call void @.omp_mapper.inner_mapper(ptr %{{.*}}, ptr %[[ELEM2]], ptr %[[ELEM2]], i64 4, i64 %{{.*}}, ptr null)
