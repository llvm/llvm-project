// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Even without debug info (no -g), the host-side __tgt_target_kernel launch
// should carry the target region's source location, taken from the op's own
// MLIR location, rather than the default ";unknown;unknown;0;0;;" identifier.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !llvm.ptr) {
      %3 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %3, %arg0 : i32, !llvm.ptr
      omp.terminator
    } loc("kernel.f90":7:3)
    llvm.return
  }
}

// CHECK: @[[SRCLOC:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";kernel.f90;{{[^;]*}};7;3;;\00"
// CHECK: @[[IDENT:[0-9]+]] = private unnamed_addr constant %struct.ident_t {{.*}}ptr @[[SRCLOC]] }
// CHECK: call i32 @__tgt_target_kernel(ptr @[[IDENT]],
