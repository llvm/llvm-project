// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// A declare_mapper map clause with an iterator modifier yields one
// bounds-described omp.map.info per iteration: the array base stays in
// var_ptr and bounds select the mapped element, the same shape used for an
// ordinary array-section map. Each iteration is pushed at runtime via
// __tgt_push_mapper_component since the trip count is not statically known.
//
// As for statically-known members, the iterated elements are grouped under a
// parent component for the mapper variable via MEMBER_OF: a storage-only
// component covering the variable is pushed first, and each element is pushed
// with the array base as its base pointer, MEMBER_OF(1) in its map type, and
// a size derived from its bounds.

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
// CHECK: %[[NUM:.*]] = call i64 @__tgt_mapper_num_components(ptr %{{.*}})
// CHECK: %[[SHIFTED:.*]] = shl i64 %[[NUM]], 48
//
// Storage-only parent component covering the whole mapper variable (40 bytes).
// CHECK: %omp.maptype = phi i64 [ 0, %omp.type.alloc ], [ 0, %omp.type.to ], [ 0, %omp.type.from ], [ 0, %omp.type.to.else ]
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %omp.arraymap.ptrcurrent, ptr %omp.arraymap.ptrcurrent, i64 40, i64 %omp.maptype.with.modifiers, ptr @{{.*}})
//
// CHECK: omp_mapper.iterator.cond:
// CHECK: %[[CMP:.*]] = icmp ult i64 %{{.*}}, 10
// CHECK: br i1 %[[CMP]], label %omp_mapper.iterator.body, label %omp_mapper.iterator.exit
//
// The field GEP (struct index 0) is computed once per iteration, then the
// bounds' lower_bound selects the element within it via an array GEP. The
// size is (ub - lb + 1) * sizeof(i32) and the map type carries MEMBER_OF(1)
// (1 << 48) | TOFROM (3) shifted by the number of pre-existing components.
// CHECK: omp_mapper.iterator.body:
// CHECK: %[[FIELD:.*]] = getelementptr %mapper_type, ptr %omp.arraymap.ptrcurrent, i32 0, i32 0
// CHECK: %[[ELEM:.*]] = getelementptr inbounds [10 x i32], ptr %[[FIELD]], i64 0, i64 %[[IV:.*]]
// CHECK: %[[EXT:.*]] = sub i64 %[[IV]], %[[IV]]
// CHECK: %[[CNT:.*]] = add i64 %[[EXT]], 1
// CHECK: %[[CNT1:.*]] = mul i64 1, %[[CNT]]
// CHECK: %[[SIZE:.*]] = mul i64 %[[CNT1]], 4
// CHECK: add nuw i64 281474976710659, %[[SHIFTED]]
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[FIELD]], ptr %[[ELEM]], i64 %[[SIZE]], i64 %{{.*}}, ptr null)

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
// CHECK: call void @.omp_mapper.inner_mapper(ptr %{{.*}}, ptr %[[FIELD2]], ptr %[[ELEM2]], i64 %{{.*}}, i64 %{{.*}}, ptr null)

// -----

// The per-iteration size comes from the bounds, so a multi-dimensional
// element (one bound per dimension) and a section (ub > lb) are sized
// correctly: 1 and 2 elements respectively, times sizeof(i32).

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper_multidim_and_section : !llvm.struct<"s", (array<4 x array<3 x i32>>, array<8 x i32>)> {
  ^bb0(%arg: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    // v%a(2, i) for i = 1..4 (0-based: a(1, iv))
    %it0 = omp.iterator(%iv: i64) = (%c0 to %c3 step %c1) {
      %field = llvm.getelementptr %arg[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"s", (array<4 x array<3 x i32>>, array<8 x i32>)>
      %b0 = omp.map.bounds lower_bound(%c1 : i64) upper_bound(%c1 : i64) extent(%c3 : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %b1 = omp.map.bounds lower_bound(%iv : i64) upper_bound(%iv : i64) extent(%c2 : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %m = omp.map.info var_ptr(%field : !llvm.ptr, !llvm.array<4 x array<3 x i32>>) map_clauses(tofrom) capture(ByRef) bounds(%b0, %b1) name("") -> !llvm.ptr
      omp.yield(%m : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    // v%c(2*j-1 : 2*j) for j = 1..4 (0-based: c(2*iv : 2*iv+1))
    %it1 = omp.iterator(%iv: i64) = (%c0 to %c3 step %c1) {
      %field = llvm.getelementptr %arg[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"s", (array<4 x array<3 x i32>>, array<8 x i32>)>
      %lb = llvm.mul %iv, %c2 : i64
      %ub = llvm.add %lb, %c1 : i64
      %b = omp.map.bounds lower_bound(%lb : i64) upper_bound(%ub : i64) extent(%c2 : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %m = omp.map.info var_ptr(%field : !llvm.ptr, !llvm.array<8 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%b) name("") -> !llvm.ptr
      omp.yield(%m : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    omp.declare_mapper.info map_iterated(%it0, %it1 : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>)
  }

  llvm.func @target_data_multidim_and_section(%addr : !llvm.ptr) {
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"s", (array<4 x array<3 x i32>>, array<8 x i32>)>) map_clauses(tofrom) capture(ByRef) mapper(@mapper_multidim_and_section) name("") -> !llvm.ptr
    omp.target_data map_entries(%map : !llvm.ptr) {}
    llvm.return
  }
}

// CHECK-LABEL: define internal void @.omp_mapper.mapper_multidim_and_section
// Storage parent covers the whole 80-byte struct.
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %omp.arraymap.ptrcurrent, ptr %omp.arraymap.ptrcurrent, i64 80, i64 %{{.*}}, ptr @{{.*}})
//
// a(2, i): two bounds, each with ub == lb, so (1 * 1) * 4 bytes. The constant
// first-dimension factor folds away; the innermost element size (i32) is
// used, not the size of the inner array.
// CHECK: omp_mapper.iterator.body:
// CHECK: %[[A:.*]] = getelementptr %s, ptr %omp.arraymap.ptrcurrent, i32 0, i32 0
// CHECK: %[[AELEM:.*]] = getelementptr inbounds [4 x [3 x i32]], ptr %[[A]], i64 0, i64 %[[AIV:.*]], i64 1
// CHECK: %[[AEXT:.*]] = sub i64 %[[AIV]], %[[AIV]]
// CHECK: %[[ACNT:.*]] = add i64 %[[AEXT]], 1
// CHECK: %[[ACNT1:.*]] = mul i64 1, %[[ACNT]]
// CHECK: %[[ASIZE:.*]] = mul i64 %[[ACNT1]], 4
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[A]], ptr %[[AELEM]], i64 %[[ASIZE]], i64 %{{.*}}, ptr null)
//
// c(2*j-1 : 2*j): one bound with ub = lb + 1, so (ub - lb + 1) * 4 bytes.
// CHECK: omp_mapper.iterator.body{{[0-9]+}}:
// CHECK: %[[C:.*]] = getelementptr %s, ptr %omp.arraymap.ptrcurrent, i32 0, i32 1
// CHECK: %[[CELEM:.*]] = getelementptr inbounds [8 x i32], ptr %[[C]], i64 0, i64 %[[CLB:.*]]
// CHECK: %[[CEXT:.*]] = sub i64 %{{.*}}, %[[CLB]]
// CHECK: %[[CCNT:.*]] = add i64 %[[CEXT]], 1
// CHECK: %[[CCNT1:.*]] = mul i64 1, %[[CCNT]]
// CHECK: %[[CSIZE:.*]] = mul i64 %[[CCNT1]], 4
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[C]], ptr %[[CELEM]], i64 %[[CSIZE]], i64 %{{.*}}, ptr null)

// -----

// When the mapper maps other members of the variable but not the variable
// itself, the synthesized storage parent for the iterated elements must be
// the first component: the runtime maps components in order and rejects a
// later one that extends an earlier mapping. The static members' MEMBER_OF
// then refers to their own (partial) parent at index 1, and the iterated
// elements' MEMBER_OF to the storage parent at index 0.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper_members_and_iterator : !llvm.struct<"s3", (i32, array<4 x i32>, i32)> {
  ^bb0(%arg: !llvm.ptr):
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %tag = llvm.getelementptr %arg[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"s3", (i32, array<4 x i32>, i32)>
    %b = llvm.getelementptr %arg[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"s3", (i32, array<4 x i32>, i32)>
    %mtag = omp.map.info var_ptr(%tag : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    %mb = omp.map.info var_ptr(%b : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    %parent = omp.map.info var_ptr(%arg : !llvm.ptr, !llvm.struct<"s3", (i32, array<4 x i32>, i32)>) map_clauses(storage) capture(ByRef) members(%mtag, %mb : [0], [2] : !llvm.ptr, !llvm.ptr) name("") partial_map(true) -> !llvm.ptr
    %it = omp.iterator(%iv: i64) = (%c0 to %c3 step %c1) {
      %field = llvm.getelementptr %arg[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"s3", (i32, array<4 x i32>, i32)>
      %bounds = omp.map.bounds lower_bound(%iv : i64) upper_bound(%iv : i64) extent(%c3 : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %m = omp.map.info var_ptr(%field : !llvm.ptr, !llvm.array<4 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%bounds) name("") -> !llvm.ptr
      omp.yield(%m : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    omp.declare_mapper.info map_entries(%parent, %mtag, %mb : !llvm.ptr, !llvm.ptr, !llvm.ptr) map_iterated(%it : !omp.iterated<!llvm.ptr>)
  }

  llvm.func @target_data_members_and_iterator(%addr : !llvm.ptr) {
    %map = omp.map.info var_ptr(%addr : !llvm.ptr, !llvm.struct<"s3", (i32, array<4 x i32>, i32)>) map_clauses(tofrom) capture(ByRef) mapper(@mapper_members_and_iterator) name("") -> !llvm.ptr
    omp.target_data map_entries(%map : !llvm.ptr) {}
    llvm.return
  }
}

// CHECK-LABEL: define internal void @.omp_mapper.mapper_members_and_iterator
// CHECK: %[[NUM3:.*]] = call i64 @__tgt_mapper_num_components(ptr %{{.*}})
// CHECK: %[[SHIFTED3:.*]] = shl i64 %[[NUM3]], 48
// Storage parent first (whole 24-byte struct) ...
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %omp.arraymap.ptrcurrent, ptr %omp.arraymap.ptrcurrent, i64 24, i64 %{{.*}}, ptr @{{.*}})
// ... then the static members' partial parent (base = struct, begin = first member) ...
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %omp.arraymap.ptrcurrent, ptr %[[TAG:.*]], i64 %{{.*}}, i64 %{{.*}}, ptr @{{.*}})
// ... the static members with MEMBER_OF(2) = 2 << 48 | TOFROM ...
// CHECK: add nuw i64 562949953421315, %[[SHIFTED3]]
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %[[TAG]], ptr %[[TAG]], i64 4, i64 %{{.*}}, ptr @{{.*}})
// CHECK: add nuw i64 562949953421315, %[[SHIFTED3]]
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 4, i64 %{{.*}}, ptr @{{.*}})
// ... and the iterated elements with MEMBER_OF(1) = 1 << 48 | TOFROM.
// CHECK: omp_mapper.iterator.body:
// CHECK: add nuw i64 281474976710659, %[[SHIFTED3]]
// CHECK: call void @__tgt_push_mapper_component(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, ptr null)
