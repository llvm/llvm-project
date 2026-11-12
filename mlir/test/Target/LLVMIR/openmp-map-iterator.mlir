// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Iterator modifiers on map and motion clauses of standalone data-mapping
// constructs fill runtime-sized offloading arrays, one entry per iteration.

// CHECK: @[[$A_NAME:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";a;

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  // CHECK-LABEL: define void @target_update_map_iterator(
  // CHECK-SAME:      ptr %[[ARR:[0-9]+]])
  // CHECK:         %[[STACK:.*]] = call ptr @llvm.stacksave.p0()
  // CHECK-NEXT:    %[[BASEPTRS:.*]] = alloca ptr, i64 10
  // CHECK-NEXT:    %[[PTRS:.*]] = alloca ptr, i64 10
  // CHECK-NEXT:    %[[SIZES:.*]] = alloca i64, i64 10
  // CHECK-NEXT:    %[[TYPES:.*]] = alloca i64, i64 10
  // CHECK-NEXT:    %[[MAPPERS:.*]] = alloca ptr, i64 10
  // CHECK-NEXT:    %[[NAMES:.*]] = alloca ptr, i64 10
  // CHECK:         icmp ult i64 %omp_offload.segment.iv, 10
  // CHECK:       omp_offload.segment.body:
  // CHECK:         %[[LB:omp.it.phys_iv]] = add i64 0, %{{.*}}
  // CHECK-NEXT:    %[[UB:.*]] = add i64 %[[LB]], 2
  // CHECK-NEXT:    %[[DIFF:.*]] = sub i64 %[[UB]], %[[LB]]
  // CHECK-NEXT:    %[[EXTENT:.*]] = add i64 %[[DIFF]], 1
  // CHECK-NEXT:    %[[COUNT:.*]] = mul i64 1, %[[EXTENT]]
  // CHECK-NEXT:    %[[SIZE:.*]] = mul i64 %[[COUNT]], 4
  // CHECK-NEXT:    %[[ELEM:.*]] = getelementptr inbounds [16 x i32], ptr %[[ARR]], i64 0, i64 %[[LB]]
  // CHECK-NEXT:    %[[IDX:.*]] = add i64 0, %omp_offload.segment.iv
  // CHECK-NEXT:    %[[BP:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr %[[ARR]], ptr %[[BP]]
  // CHECK-NEXT:    %[[P:.*]] = getelementptr inbounds ptr, ptr %[[PTRS]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr %[[ELEM]], ptr %[[P]]
  // CHECK-NEXT:    %[[S:.*]] = getelementptr inbounds i64, ptr %[[SIZES]], i64 %[[IDX]]
  // CHECK-NEXT:    store i64 %[[SIZE]], ptr %[[S]]
  // CHECK-NEXT:    %[[T:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 %[[IDX]]
  // CHECK-NEXT:    store i64 1, ptr %[[T]]
  // CHECK-NEXT:    %[[M:.*]] = getelementptr inbounds ptr, ptr %[[MAPPERS]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr null, ptr %[[M]]
  // CHECK-NEXT:    %[[N:.*]] = getelementptr inbounds ptr, ptr %[[NAMES]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr @[[$A_NAME]], ptr %[[N]]
  // CHECK:       omp.it.cont:
  // CHECK-NEXT:    call void @__tgt_target_data_update_mapper(ptr @{{.*}}, i64 -1, i32 10, ptr %[[BASEPTRS]], ptr %[[PTRS]], ptr %[[SIZES]], ptr %[[TYPES]], ptr %[[NAMES]], ptr null)
  // CHECK-NEXT:    call void @llvm.stackrestore.p0(ptr %[[STACK]])
  llvm.func @target_update_map_iterator(%arr: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c10 = llvm.mlir.constant(10 : i64) : i64
    %it = omp.iterator(%iv: i64) = (%c0 to %c10 step %c1) {
      %ub = llvm.add %iv, %c2 : i64
      %bounds = omp.map.bounds lower_bound(%iv : i64) upper_bound(%ub : i64) extent(%c2 : i64) stride(%c1 : i64) start_idx(%c0 : i64)
      %map = omp.map.info var_ptr(%arr : !llvm.ptr, !llvm.array<16 x i32>) map_clauses(to) capture(ByRef) bounds(%bounds) name("a") -> !llvm.ptr
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target_update map_iterated(%it : !omp.iterated<!llvm.ptr>)
    llvm.return
  }

  // The static entry comes first; the iterated entries follow it.
  // CHECK-LABEL: define void @target_enter_data_map_iterator_mixed(
  // CHECK-SAME:      ptr %[[ARR:[0-9]+]], ptr %[[SCALAR:[0-9]+]], i64 %[[LB:[0-9]+]], i64 %{{[0-9]+}})
  // CHECK:         %[[TRIPS:.*]] = trunc i65 %omp_iterator.tripcount to i64
  // CHECK-NEXT:    %[[COUNT:.*]] = mul i64 1, %[[TRIPS]]
  // CHECK-NEXT:    %[[STACK:.*]] = call ptr @llvm.stacksave.p0()
  // CHECK-NEXT:    %[[TOTAL:.*]] = add i64 1, %[[COUNT]]
  // CHECK-NEXT:    %[[BASEPTRS:.*]] = alloca ptr, i64 %[[TOTAL]]
  // CHECK-NEXT:    %[[PTRS:.*]] = alloca ptr, i64 %[[TOTAL]]
  // CHECK-NEXT:    %[[SIZES:.*]] = alloca i64, i64 %[[TOTAL]]
  // CHECK-NEXT:    %[[TYPES:.*]] = alloca i64, i64 %[[TOTAL]]
  // CHECK:         %[[BP0:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 0
  // CHECK-NEXT:    store ptr %[[SCALAR]], ptr %[[BP0]]
  // CHECK:         %[[T0:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 0
  // CHECK-NEXT:    store i64 1, ptr %[[T0]]
  // CHECK:         icmp ult i64 %omp_offload.segment.iv, %[[COUNT]]
  // CHECK:       omp_offload.segment.body:
  // CHECK:         %omp.it.phys_iv = add i64 %[[LB]], %{{.*}}
  // CHECK-NEXT:    %[[ELEM:.*]] = getelementptr i32, ptr %[[ARR]], i64 %omp.it.phys_iv
  // CHECK-NEXT:    %[[IDX:.*]] = add i64 1, %omp_offload.segment.iv
  // CHECK-NEXT:    %[[BP:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr %[[ELEM]], ptr %[[BP]]
  // CHECK-NEXT:    %[[P:.*]] = getelementptr inbounds ptr, ptr %[[PTRS]], i64 %[[IDX]]
  // CHECK-NEXT:    store ptr %[[ELEM]], ptr %[[P]]
  // CHECK-NEXT:    %[[S:.*]] = getelementptr inbounds i64, ptr %[[SIZES]], i64 %[[IDX]]
  // CHECK-NEXT:    store i64 4, ptr %[[S]]
  // CHECK-NEXT:    %[[T:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 %[[IDX]]
  // CHECK-NEXT:    store i64 5, ptr %[[T]]
  // CHECK:       omp.it.cont:
  // CHECK-NEXT:    %[[NUM:.*]] = trunc i64 %[[TOTAL]] to i32
  // CHECK-NEXT:    call void @__tgt_target_data_begin_mapper(ptr @{{.*}}, i64 -1, i32 %[[NUM]], ptr %[[BASEPTRS]], ptr %[[PTRS]], ptr %[[SIZES]], ptr %[[TYPES]],
  // CHECK-NEXT:    call void @llvm.stackrestore.p0(ptr %[[STACK]])
  llvm.func @target_enter_data_map_iterator_mixed(%arr: !llvm.ptr, %scalar: !llvm.ptr, %lb: i64, %ub: i64) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %static = omp.map.info var_ptr(%scalar : !llvm.ptr, f64) map_clauses(to) capture(ByRef) name("s") -> !llvm.ptr
    %it = omp.iterator(%iv: i64) = (%lb to %ub step %c1) {
      %elem = llvm.getelementptr %arr[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i32) map_clauses(always, to) capture(ByRef) name("") -> !llvm.ptr
      omp.yield(%map : !llvm.ptr)
    } inclusive -> !omp.iterated<!llvm.ptr>
    omp.target_enter_data map_entries(%static : !llvm.ptr) map_iterated(%it : !omp.iterated<!llvm.ptr>) {}
    llvm.return
  }

  // Each iterator fills its own loop, after the entries of the previous one.
  // The arrays only live in the then branch of the if clause.
  // CHECK-LABEL: define void @target_exit_data_map_iterators(
  // CHECK-SAME:      ptr %[[A:[0-9]+]], ptr %[[B:[0-9]+]], i1 %[[COND:[0-9]+]])
  // CHECK:         br i1 %[[COND]], label %omp_if.then, label %omp_if.else
  // CHECK:       omp_if.then:
  // CHECK-NEXT:    %[[STACK:.*]] = call ptr @llvm.stacksave.p0()
  // CHECK-NEXT:    %[[BASEPTRS:.*]] = alloca ptr, i64 8
  // CHECK-NEXT:    %[[PTRS:.*]] = alloca ptr, i64 8
  // CHECK-NEXT:    %[[SIZES:.*]] = alloca i64, i64 8
  // CHECK-NEXT:    %[[TYPES:.*]] = alloca i64, i64 8
  // CHECK:         icmp ult i64 %omp_offload.segment.iv, 6
  // CHECK:         %[[J:omp.it.phys_iv[0-9]*]] = add i64 0, %{{.*}}
  // CHECK:         %[[I:omp.it.phys_iv[0-9]*]] = add i64 0, %{{.*}}
  // CHECK-NEXT:    %[[ROW:.*]] = mul i64 %[[I]], 3
  // CHECK-NEXT:    %[[LINEAR:.*]] = add i64 %[[ROW]], %[[J]]
  // CHECK-NEXT:    %[[ELEMA:.*]] = getelementptr i32, ptr %[[A]], i64 %[[LINEAR]]
  // CHECK-NEXT:    %[[IDXA:.*]] = add i64 0, %omp_offload.segment.iv
  // CHECK-NEXT:    %[[BPA:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 %[[IDXA]]
  // CHECK-NEXT:    store ptr %[[ELEMA]], ptr %[[BPA]]
  // CHECK:         %[[TA:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 %[[IDXA]]
  // CHECK-NEXT:    store i64 2, ptr %[[TA]]
  // CHECK:         icmp ult i64 %[[IVB:omp_offload.segment.iv[0-9]+]], 2
  // CHECK:         %[[K:omp.it.phys_iv[0-9]*]] = add i64 0, %{{.*}}
  // CHECK-NEXT:    %[[ELEMB:.*]] = getelementptr i64, ptr %[[B]], i64 %[[K]]
  // CHECK-NEXT:    %[[IDXB:.*]] = add i64 6, %[[IVB]]
  // CHECK-NEXT:    %[[BPB:.*]] = getelementptr inbounds ptr, ptr %[[BASEPTRS]], i64 %[[IDXB]]
  // CHECK-NEXT:    store ptr %[[ELEMB]], ptr %[[BPB]]
  // CHECK:         %[[SB:.*]] = getelementptr inbounds i64, ptr %[[SIZES]], i64 %[[IDXB]]
  // CHECK-NEXT:    store i64 8, ptr %[[SB]]
  // CHECK-NEXT:    %[[TB:.*]] = getelementptr inbounds i64, ptr %[[TYPES]], i64 %[[IDXB]]
  // CHECK-NEXT:    store i64 8, ptr %[[TB]]
  // CHECK:         call void @__tgt_target_data_end_mapper(ptr @{{.*}}, i64 -1, i32 8, ptr %[[BASEPTRS]], ptr %[[PTRS]], ptr %[[SIZES]], ptr %[[TYPES]],
  // CHECK-NEXT:    call void @llvm.stackrestore.p0(ptr %[[STACK]])
  // CHECK-NEXT:    br label %omp_if.end
  llvm.func @target_exit_data_map_iterators(%a: !llvm.ptr, %b: !llvm.ptr, %cond: i1) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %ita = omp.iterator(%i: i64, %j: i64) = (%c0 to %c2 step %c1, %c0 to %c3 step %c1) {
      %row = llvm.mul %i, %c3 : i64
      %linear = llvm.add %row, %j : i64
      %elem = llvm.getelementptr %a[%linear] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i32) map_clauses(from) capture(ByRef) name("") -> !llvm.ptr
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    %itb = omp.iterator(%iv: i64) = (%c0 to %c2 step %c1) {
      %elem = llvm.getelementptr %b[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %map = omp.map.info var_ptr(%elem : !llvm.ptr, i64) map_clauses(delete) capture(ByRef) name("") -> !llvm.ptr
      omp.yield(%map : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>
    omp.target_exit_data if(%cond) map_iterated(%ita, %itb : !omp.iterated<!llvm.ptr>, !omp.iterated<!llvm.ptr>) {}
    llvm.return
  }
}
