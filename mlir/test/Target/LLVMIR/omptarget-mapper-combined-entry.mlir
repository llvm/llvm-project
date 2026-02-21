// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Ensure that user-defined mappers are only attached to the base entry of a
// combined parent mapping. Attaching the mapper to segment entries can invoke
// the mapper with a partial size, which is undefined behaviour.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @mapper : !llvm.struct<"S", (i32, i32)> {
  ^bb0(%arg0: !llvm.ptr):
    %field0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"S", (i32, i32)>
    %map_field0 = omp.map.info var_ptr(%field0 : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map_parent = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"S", (i32, i32)>)
        map_clauses(tofrom) capture(ByRef) members(%map_field0 : [0] : !llvm.ptr) -> !llvm.ptr
        {name = "", partial_map = true}
    omp.declare_mapper.info map_entries(%map_parent, %map_field0 : !llvm.ptr, !llvm.ptr)
  }

  llvm.func @test_mapper_combined_entries() {
    %one = llvm.mlir.constant(1 : i64) : i64
    %s = llvm.alloca %one x !llvm.struct<"S", (i32, i32)> : (i64) -> !llvm.ptr
    %field0 = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"S", (i32, i32)>
    %map_field0 = omp.map.info var_ptr(%field0 : !llvm.ptr, i32)
        map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "s.f0"}
    %map_parent = omp.map.info var_ptr(%s : !llvm.ptr, !llvm.struct<"S", (i32, i32)>)
        map_clauses(tofrom) capture(ByRef) mapper(@mapper) members(%map_field0 : [0] : !llvm.ptr) -> !llvm.ptr
        {name = "s"}
    omp.target map_entries(%map_parent -> %arg0, %map_field0 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @test_mapper_combined_entries
// CHECK: %[[MAPPERS:.*offload_mappers.*]] = alloca [4 x ptr]
// CHECK: %[[MAPPER0:.*]] = getelementptr inbounds [4 x ptr], ptr %[[MAPPERS]], i64 0, i64 0
// CHECK: store ptr @.omp_mapper.mapper, ptr %[[MAPPER0]]
// CHECK: %[[MAPPER1:.*]] = getelementptr inbounds [4 x ptr], ptr %[[MAPPERS]], i64 0, i64 1
// CHECK: store ptr null, ptr %[[MAPPER1]]
// CHECK: %[[MAPPER2:.*]] = getelementptr inbounds [4 x ptr], ptr %[[MAPPERS]], i64 0, i64 2
// CHECK: store ptr null, ptr %[[MAPPER2]]
