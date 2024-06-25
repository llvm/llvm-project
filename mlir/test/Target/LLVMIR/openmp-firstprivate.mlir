// Test code-gen for `omp.parallel` ops with delayed privatizers (i.e. using
// `omp.private` ops).

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @parallel_op_firstprivate(%arg0: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

omp.private {type = firstprivate} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %0, %arg1 : f32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

// CHECK-LABEL: @parallel_op_firstprivate
// CHECK-SAME: (ptr %[[ORIG:.*]]) {
// CHECK: %[[OMP_PAR_ARG:.*]] = alloca { ptr }, align 8
// CHECK: %[[ORIG_GEP:.*]] = getelementptr { ptr }, ptr %[[OMP_PAR_ARG]], i32 0, i32 0
// CHECK: store ptr %[[ORIG]], ptr %[[ORIG_GEP]], align 8
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @parallel_op_firstprivate..omp_par, ptr %[[OMP_PAR_ARG]])
// CHECK: }

// CHECK-LABEL: void @parallel_op_firstprivate..omp_par
// CHECK-SAME: (ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %[[ARG:.*]])
// CHECK: %[[ORIG_PTR_PTR:.*]] = getelementptr { ptr }, ptr %[[ARG]], i32 0, i32 0
// CHECK: %[[ORIG_PTR:.*]] = load ptr, ptr %[[ORIG_PTR_PTR]], align 8

// Check that the privatizer alloc region was inlined properly.
// CHECK: %[[PRIV_ALLOC:.*]] = alloca float, align 4

// Check that the privatizer copy region was inlined properly.

// CHECK: %[[ORIG_VAL:.*]] = load float, ptr %[[ORIG_PTR]], align 4
// CHECK: store float %[[ORIG_VAL]], ptr %[[PRIV_ALLOC]], align 4
// CHECK-NEXT: br

// Check that the privatized value is used (rather than the original one).
// CHECK: load float, ptr %[[PRIV_ALLOC]], align 4
// CHECK: }

// -----

llvm.func @parallel_op_firstprivate_multi_block(%arg0: !llvm.ptr) {
  omp.parallel private(@multi_block.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @parallel_op_firstprivate_multi_block..omp_par
// CHECK: omp.par.entry:
// CHECK:  %[[ORIG_PTR_PTR:.*]] = getelementptr { ptr }, ptr %{{.*}}, i32 0, i32 0
// CHECK:  %[[ORIG_PTR:.*]] = load ptr, ptr %[[ORIG_PTR_PTR]], align 8
// CHECK:   br label %[[PRIV_BB1:.*]]

// CHECK: [[PRIV_BB1]]:
// The 1st `alloc` block directly branches to the 2nd `alloc` block since the
// only insruction is `llvm.mlir.constant` which gets translated to compile-time
// constant in LLVM IR.
// CHECK-NEXT: br label %[[PRIV_BB2:.*]]

// CHECK: [[PRIV_BB2]]:
// CHECK-NEXT: %[[C1:.*]] = phi i32 [ 1, %[[PRIV_BB1]] ]
// CHECK-NEXT: %[[PRIV_ALLOC:.*]] = alloca float, i32 %[[C1]], align 4
// The entry block of the `copy` region is merged into the exit block of the
// `alloc` region. So check for that.
// CHECK-NEXT: %[[ORIG_VAL:.*]] = load float, ptr %[[ORIG_PTR]], align 4
// CHECK-NEXT: br label %[[PRIV_BB3:.*]]

// Check contents of the 2nd block in the `copy` region.
// CHECK: [[PRIV_BB3]]:
// CHECK-NEXT: %[[ORIG_VAL2:.*]] = phi float [ %[[ORIG_VAL]], %[[PRIV_BB2]] ]
// CHECK-NEXT: %[[PRIV_ALLOC2:.*]] = phi ptr [ %[[PRIV_ALLOC]], %[[PRIV_BB2]] ]
// CHECK-NEXT: store float %[[ORIG_VAL2]], ptr %[[PRIV_ALLOC2]], align 4
// CHECK-NEXT: br label %[[PRIV_CONT:.*]]

// Check that the privatizer's continuation block yileds the private clone's
// address.
// CHECK: [[PRIV_CONT]]:
// CHECK-NEXT:   %[[PRIV_ALLOC3:.*]] = phi ptr [ %[[PRIV_ALLOC2]], %[[PRIV_BB3]] ]
// CHECK-NEXT:   br label %[[PAR_REG:.*]]

// Check that the body of the parallel region loads from the private clone.
// CHECK: [[PAR_REG]]:
// CHECK:        %{{.*}} = load float, ptr %[[PRIV_ALLOC3]], align 4

omp.private {type = firstprivate} @multi_block.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^bb1(%c1 : i32)

^bb1(%arg1: i32):
  %0 = llvm.alloca %arg1 x f32 : (i32) -> !llvm.ptr
  omp.yield(%0 : !llvm.ptr)

} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.br ^bb1(%0, %arg1 : f32, !llvm.ptr)

^bb1(%arg2: f32, %arg3: !llvm.ptr):
  llvm.store %arg2, %arg3 : f32, !llvm.ptr
  omp.yield(%arg3 : !llvm.ptr)
}
