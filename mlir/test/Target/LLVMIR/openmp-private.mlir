// Test code-gen for `omp.parallel` ops with delayed privatizers (i.e. using
// `omp.private` ops).

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @parallel_op_1_private(%arg0: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = ptr.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: @parallel_op_1_private
// CHECK-SAME: (ptr %[[ORIG:.*]]) {
// CHECK: %[[OMP_PAR_ARG:.*]] = alloca { ptr }, align 8
// CHECK: %[[ORIG_GEP:.*]] = getelementptr { ptr }, ptr %[[OMP_PAR_ARG]], i32 0, i32 0
// CHECK: store ptr %[[ORIG]], ptr %[[ORIG_GEP]], align 8
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @parallel_op_1_private..omp_par, ptr %[[OMP_PAR_ARG]])
// CHECK: }

// CHECK-LABEL: void @parallel_op_1_private..omp_par
// CHECK-SAME: (ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %[[ARG:.*]])
// CHECK: %[[ORIG_PTR_PTR:.*]] = getelementptr { ptr }, ptr %[[ARG]], i32 0, i32 0
// CHECK: %[[ORIG_PTR:.*]] = load ptr, ptr %[[ORIG_PTR_PTR]], align 8

// Check that the privatizer alloc region was inlined properly.
// CHECK: %[[PRIV_ALLOC:.*]] = alloca float, align 4
// CHECK: %[[ORIG_VAL:.*]] = load float, ptr %[[ORIG_PTR]], align 4
// CHECK: store float %[[ORIG_VAL]], ptr %[[PRIV_ALLOC]], align 4
// CHECK-NEXT: br

// Check that the privatized value is used (rather than the original one).
// CHECK: load float, ptr %[[PRIV_ALLOC]], align 4
// CHECK: }

llvm.func @parallel_op_2_privates(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr, @y.privatizer %arg1 -> %arg3 : !llvm.ptr) {
    %0 = ptr.load %arg2 : !llvm.ptr -> f32
    %1 = ptr.load %arg3 : !llvm.ptr -> i32
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: @parallel_op_2_privates
// CHECK-SAME: (ptr %[[ORIG1:.*]], ptr %[[ORIG2:.*]]) {
// CHECK: %[[OMP_PAR_ARG:.*]] = alloca { ptr, ptr }, align 8
// CHECK: %[[ORIG1_GEP:.*]] = getelementptr { ptr, ptr }, ptr %[[OMP_PAR_ARG]], i32 0, i32 0
// CHECK: store ptr %[[ORIG1]], ptr %[[ORIG1_GEP]], align 8
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @parallel_op_2_privates..omp_par, ptr %[[OMP_PAR_ARG]])
// CHECK: }

// CHECK-LABEL: void @parallel_op_2_privates..omp_par
// CHECK-SAME: (ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %[[ARG:.*]])
// CHECK: %[[ORIG1_PTR_PTR:.*]] = getelementptr { ptr, ptr }, ptr %[[ARG]], i32 0, i32 0
// CHECK: %[[ORIG1_PTR:.*]] = load ptr, ptr %[[ORIG1_PTR_PTR]], align 8
// CHECK: %[[ORIG2_PTR_PTR:.*]] = getelementptr { ptr, ptr }, ptr %[[ARG]], i32 0, i32 1
// CHECK: %[[ORIG2_PTR:.*]] = load ptr, ptr %[[ORIG2_PTR_PTR]], align 8

// Check that the privatizer alloc region was inlined properly.
// CHECK: %[[PRIV1_ALLOC:.*]] = alloca float, align 4
// CHECK: %[[ORIG1_VAL:.*]] = load float, ptr %[[ORIG1_PTR]], align 4
// CHECK: store float %[[ORIG1_VAL]], ptr %[[PRIV1_ALLOC]], align 4
// CHECK: %[[PRIV2_ALLOC:.*]] = alloca i32, align 4
// CHECK: %[[ORIG2_VAL:.*]] = load i32, ptr %[[ORIG2_PTR]], align 4
// CHECK: store i32 %[[ORIG2_VAL]], ptr %[[PRIV2_ALLOC]], align 4
// CHECK-NEXT: br

// Check that the privatized value is used (rather than the original one).
// CHECK: load float, ptr %[[PRIV1_ALLOC]], align 4
// CHECK: load i32, ptr %[[PRIV2_ALLOC]], align 4
// CHECK: }

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  %1 = ptr.load %arg0 : !llvm.ptr -> f32
  ptr.store %1, %0 : f32, !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

omp.private {type = private} @y.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %1 = ptr.load %arg0 : !llvm.ptr -> i32
  ptr.store %1, %0 : i32, !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

// -----

llvm.func @parallel_op_private_multi_block(%arg0: !llvm.ptr) {
  omp.parallel private(@multi_block.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = ptr.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @parallel_op_private_multi_block..omp_par
// CHECK: omp.par.entry:
// CHECK:  %[[ORIG_PTR_PTR:.*]] = getelementptr { ptr }, ptr %{{.*}}, i32 0, i32 0
// CHECK:  %[[ORIG_PTR:.*]] = load ptr, ptr %[[ORIG_PTR_PTR]], align 8
// CHECK:   br label %[[PRIV_BB1:.*]]

// Check contents of the first block in the `alloc` region.
// CHECK: [[PRIV_BB1]]:
// CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = alloca float, align 4
// CHECK-NEXT:   br label %[[PRIV_BB2:.*]]

// Check contents of the second block in the `alloc` region.
// CHECK: [[PRIV_BB2]]:
// CHECK-NEXT:   %[[ORIG_PTR2:.*]] = phi ptr [ %[[ORIG_PTR]], %[[PRIV_BB1]] ]
// CHECK-NEXT:   %[[PRIV_ALLOC2:.*]] = phi ptr [ %[[PRIV_ALLOC]], %[[PRIV_BB1]] ]
// CHECK-NEXT:   %[[ORIG_VAL:.*]] = load float, ptr %[[ORIG_PTR2]], align 4
// CHECK-NEXT:   store float %[[ORIG_VAL]], ptr %[[PRIV_ALLOC2]], align 4
// CHECK-NEXT:   br label %[[PRIV_CONT:.*]]

// Check that the privatizer's continuation block yileds the private clone's
// address.
// CHECK: [[PRIV_CONT]]:
// CHECK-NEXT:   %[[PRIV_ALLOC3:.*]] = phi ptr [ %[[PRIV_ALLOC2]], %[[PRIV_BB2]] ]
// CHECK-NEXT:   br label %[[PAR_REG:.*]]

// Check that the body of the parallel region loads from the private clone.
// CHECK: [[PAR_REG]]:
// CHECK:        %{{.*}} = load float, ptr %[[PRIV_ALLOC3]], align 4

omp.private {type = private} @multi_block.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  llvm.br ^bb1(%arg0, %0 : !llvm.ptr, !llvm.ptr)

^bb1(%arg1: !llvm.ptr, %arg2: !llvm.ptr):
  %1 = ptr.load %arg1 : !llvm.ptr -> f32
  ptr.store %1, %arg2 : f32, !llvm.ptr
  omp.yield(%arg2 : !llvm.ptr)
}
