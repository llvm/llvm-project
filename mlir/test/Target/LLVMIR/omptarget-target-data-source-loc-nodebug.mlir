// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Without debug info, the target data begin/end mappers must still carry the
// region's source location (built from the omp.target_data op's own MLIR
// location) instead of the default ";unknown;unknown;0;0;;". Requires
// target_triples so an offloading entry is generated.

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @_QPtdata() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target_data map_entries(%2 : !llvm.ptr) {
      %3 = llvm.mlir.constant(99 : i32) : i32
      llvm.store %3, %1 : i32, !llvm.ptr
      omp.terminator
    } loc(#loc1)
    llvm.return
  }
}
#loc1 = loc("target_data.f90":7:9)

// The begin and end mappers of the same region share one identifier that
// references the real file/line source-location string.
// CHECK: @[[SRC:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";target_data.f90;{{[^;]*}};7;9;;\00"
// CHECK: @[[IDENT:[0-9]+]] = private unnamed_addr constant %struct.ident_t {{.*}}ptr @[[SRC]] }
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @[[IDENT]],
// CHECK: call void @__tgt_target_data_end_mapper(ptr @[[IDENT]],
