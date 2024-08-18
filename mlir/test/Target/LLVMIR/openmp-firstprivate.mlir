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

// -----

// Verifies fix for https://github.com/llvm/llvm-project/issues/102935.
//
// The issue happens since we previously failed to match MLIR values to their
// corresponding LLVM values in some cases (e.g. char strings with non-const
// length).
llvm.func @non_const_len_char_test(%n: !llvm.ptr {fir.bindc_name = "n"}) {
  %n_val = llvm.load %n : !llvm.ptr -> i64
  %orig_alloc = llvm.alloca %n_val x i8 {bindc_name = "str"} : (i64) -> !llvm.ptr
  %orig_val = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %orig_val_init = llvm.insertvalue %orig_alloc, %orig_val[0] : !llvm.struct<(ptr, i64)>
  omp.parallel private(@non_const_len_char %orig_val_init -> %priv_arg : !llvm.struct<(ptr, i64)>) {
    %dummy = llvm.extractvalue %priv_arg[0] : !llvm.struct<(ptr, i64)>
    omp.terminator
  }
  llvm.return
}

omp.private {type = firstprivate} @non_const_len_char : !llvm.struct<(ptr, i64)> alloc {
^bb0(%orig_val: !llvm.struct<(ptr, i64)>):
  %str_len = llvm.extractvalue %orig_val[1] : !llvm.struct<(ptr, i64)>
  %priv_alloc = llvm.alloca %str_len x i8 {bindc_name = "str", pinned} : (i64) -> !llvm.ptr
  %priv_val = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %priv_val_init = llvm.insertvalue %priv_alloc, %priv_val[0] : !llvm.struct<(ptr, i64)>
  omp.yield(%priv_val_init : !llvm.struct<(ptr, i64)>)
} copy {
^bb0(%orig_val: !llvm.struct<(ptr, i64)>, %priv_val: !llvm.struct<(ptr, i64)>):
  llvm.call @foo() : () -> ()
  omp.yield(%priv_val : !llvm.struct<(ptr, i64)>)
}

llvm.func @foo()

// CHECK-LABEL: @non_const_len_char_test..omp_par({{.*}})
// CHECK-NEXT:    omp.par.entry:
// Verify that we found the privatizer by checking that we properly inlined the
// bodies of the alloc and copy regions.
// CHECK:         %[[STR_LEN:.*]] = extractvalue { ptr, i64 } %{{.*}}, 1
// CHECK:         %{{.*}} = alloca i8, i64 %[[STR_LEN]], align 1
// CHECK:         call void @foo()
