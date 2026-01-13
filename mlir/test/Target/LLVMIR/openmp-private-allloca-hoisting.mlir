// Tests that static alloca's in `omp.private ... init` regions are hoisted to
// the parent construct's alloca IP.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
llvm.func @foo1()
llvm.func @foo2()
llvm.func @foo3()
llvm.func @foo4()

omp.private {type = private} @multi_block.privatizer : f32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %alloca1 = llvm.alloca %0 x !llvm.struct<(i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>

  %1 = llvm.load %arg0 : !llvm.ptr -> f32

  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %cond1 = llvm.icmp "eq" %c1, %c2 : i32
  llvm.cond_br %cond1, ^bb1, ^bb2

^bb1:
  llvm.call @foo1() : () -> ()
  llvm.br ^bb3

^bb2:
  llvm.call @foo2() : () -> ()
  llvm.br ^bb3

^bb3:
  llvm.store %1, %arg1 : f32, !llvm.ptr

  omp.yield(%arg1 : !llvm.ptr)
}

omp.private {type = private} @multi_block.privatizer2 : f32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %alloca1 = llvm.alloca %0 x !llvm.struct<(ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>

  %1 = llvm.load %arg0 : !llvm.ptr -> f32

  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %cond1 = llvm.icmp "eq" %c1, %c2 : i32
  llvm.cond_br %cond1, ^bb1, ^bb2

^bb1:
  llvm.call @foo3() : () -> ()
  llvm.br ^bb3

^bb2:
  llvm.call @foo4() : () -> ()
  llvm.br ^bb3

^bb3:
  llvm.store %1, %arg1 : f32, !llvm.ptr

  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @parallel_op_private_multi_block(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %arg0_map = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.ptr)
        map_clauses(is_device_ptr) capture(ByRef) -> !llvm.ptr {name = ""}
  %arg1_map = omp.map.info var_ptr(%arg1 : !llvm.ptr, !llvm.ptr)
        map_clauses(is_device_ptr) capture(ByRef) -> !llvm.ptr {name = ""}

  omp.target map_entries(%arg0_map -> %arg0_arg, %arg1_map -> %arg1_arg : !llvm.ptr, !llvm.ptr) {
  omp.parallel private(@multi_block.privatizer %arg0_arg -> %arg2,
                       @multi_block.privatizer2 %arg1_arg -> %arg3 : !llvm.ptr, !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    %1 = llvm.load %arg3 : !llvm.ptr -> f32
    omp.terminator
  }
    omp.terminator
  }
  llvm.return
}
}

// CHECK: call void @__kmpc_parallel_60(ptr addrspacecast (ptr addrspace(1) @3 to ptr), i32 %omp_global_thread_num, i32 1, i32 -1, i32 -1, ptr @[[OUTLINED_FN:[^[:space:]]+]], {{.*}})

// CHECK: define internal void @[[OUTLINED_FN]]({{.*}}) {{.*}} {
// CHECK: omp.par.entry:
// Varify that both allocas were hoisted to the parallel region's entry block.
// CHECK:        %{{.*}} = alloca { i64 }, align 8
// CHECK-NEXT:   %{{.*}} = alloca { ptr }, align 8
// CHECK-NEXT:   br label %omp.region.after_alloca1
// CHECK: omp.region.after_alloca1:
// CHECK: }
