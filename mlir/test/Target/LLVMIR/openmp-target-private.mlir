// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @simple_var.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @target_map_single_private() attributes {fir.internal_name = "_QPtarget_map_single_private"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %4, %3 : i32, !llvm.ptr
  %5 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%5 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1 : !llvm.ptr) {
    %6 = llvm.mlir.constant(10 : i32) : i32
    %7 = llvm.load %arg0 : !llvm.ptr -> i32
    %8 = llvm.add %7, %6 : i32
    llvm.store %8, %arg1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @__omp_offloading_
// CHECK-NOT: define {{.*}}
// CHECK: %[[PRIV_ALLOC:.*]] = alloca i32, i64 1, align 4
// CHECK: %[[ADD:.*]] = add i32 {{.*}}, 10
// CHECK: store i32 %[[ADD]], ptr %[[PRIV_ALLOC]], align 4

omp.private {type = private} @n.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "n", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}
llvm.func @target_map_2_privates() attributes {fir.internal_name = "_QPtarget_map_2_privates"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "simple_var"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x f32 {bindc_name = "n"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %6, %5 : i32, !llvm.ptr
  %7 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "a"}
  omp.target map_entries(%7 -> %arg0 : !llvm.ptr) private(@simple_var.privatizer %1 -> %arg1, @n.privatizer %3 -> %arg2 : !llvm.ptr, !llvm.ptr) {
    %8 = llvm.mlir.constant(1.100000e+01 : f32) : f32
    %9 = llvm.mlir.constant(10 : i32) : i32
    %10 = llvm.load %arg0 : !llvm.ptr -> i32
    %11 = llvm.add %10, %9 : i32
    llvm.store %11, %arg1 : i32, !llvm.ptr
    %12 = llvm.load %arg1 : !llvm.ptr -> i32
    %13 = llvm.sitofp %12 : i32 to f32
    %14 = llvm.fadd %13, %8  {fastmathFlags = #llvm.fastmath<contract>} : f32
    llvm.store %14, %arg2 : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}


// CHECK: define internal void @__omp_offloading_
// CHECK: %[[PRIV_I32_ALLOC:.*]] = alloca i32, i64 1, align 4
// CHECK: %[[PRIV_FLOAT_ALLOC:.*]] = alloca float, i64 1, align 4
// CHECK: %[[ADD_I32:.*]] = add i32 {{.*}}, 10
// CHECK: store i32 %[[ADD_I32]], ptr %[[PRIV_I32_ALLOC]], align 4
// CHECK: %[[LOAD_I32_AGAIN:.*]] = load i32, ptr %[[PRIV_I32_ALLOC]], align 4
// CHECK: %[[CAST_TO_FLOAT:.*]] = sitofp i32 %[[LOAD_I32_AGAIN]] to float
// CHECK: %[[ADD_FLOAT:.*]] = fadd contract float %[[CAST_TO_FLOAT]], 1.100000e+01
// CHECK: store float %[[ADD_FLOAT]], ptr %[[PRIV_FLOAT_ALLOC]], align 4

// An entirely artifical privatizer that is meant to check multi-block
// privatizers. The idea here is to prove that we set the correct
// insertion points for the builder when generating, first, LLVM IR for the
// privatizer and then for the actual target region.
omp.private {type = private} @multi_block.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^bb1(%c1 : i32)

^bb1(%arg1: i32):
  %0 = llvm.alloca %arg1 x f32 : (i32) -> !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

llvm.func @target_op_private_multi_block(%arg0: !llvm.ptr) {
  omp.target private(@multi_block.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}
// CHECK: define internal void @__omp_offloading_
// CHECK: %[[ONE:.*]] = phi i32 [ 1, {{.*}} ]
// CHECK: %[[PRIV_ALLOC:.*]] = alloca float, i32 %[[ONE]], align 4
// CHECK: %[[PHI_ALLOCA:.*]]  = phi ptr [ %[[PRIV_ALLOC]], {{.*}} ]
// CHECK: %[[RESULT:.*]] = load float, ptr %[[PHI_ALLOCA]], align 4
