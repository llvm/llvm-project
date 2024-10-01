// Test code-gen for `omp.parallel` ops with delayed privatizers (i.e. using
// `omp.private` ops).

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @parallel_op_1_private(%arg0: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
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
  omp.parallel private(@x.privatizer %arg0 -> %arg2, @y.privatizer %arg1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    %1 = llvm.load %arg3 : !llvm.ptr -> i32
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
  %1 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %1, %0 : f32, !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

omp.private {type = private} @y.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %1 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %1, %0 : i32, !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

// -----

llvm.func @parallel_op_private_multi_block(%arg0: !llvm.ptr) {
  omp.parallel private(@multi_block.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
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
  %1 = llvm.load %arg1 : !llvm.ptr -> f32
  llvm.store %1, %arg2 : f32, !llvm.ptr
  omp.yield(%arg2 : !llvm.ptr)
}

// Tests fix for Fujitsu test suite test: 0007_0019.f90: the
// `llvm.mlir.addressof` op needs access to the parent module when lowering
// from the LLVM dialect to LLVM IR. If such op is used inside an `omp.private`
// op instance that was not created/cloned inside the module, we would get a
// seg fault due to trying to access a null pointer.

// CHECK-LABEL: define internal void @lower_region_with_addressof..omp_par
// CHECK:         omp.par.region:
// CHECK:           br label %[[PAR_REG_BEG:.*]]
// CHECK:         [[PAR_REG_BEG]]:
// CHECK:           %[[PRIVATIZER_GEP:.*]] = getelementptr double, ptr @_QQfoo, i64 111
// CHECK:           call void @bar(ptr %[[PRIVATIZER_GEP]])
// CHECK:           call void @bar(ptr getelementptr (double, ptr @_QQfoo, i64 222))
llvm.func @lower_region_with_addressof() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f64 {bindc_name = "u1"} : (i64) -> !llvm.ptr
  omp.parallel private(@_QFlower_region_with_addressof_privatizer %1 -> %arg0 : !llvm.ptr) {
    %c0 = llvm.mlir.constant(111 : i64) : i64
    %2 = llvm.getelementptr %arg0[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.call @bar(%2) : (!llvm.ptr) -> ()

    %c1 = llvm.mlir.constant(222 : i64) : i64
    %3 = llvm.mlir.addressof @_QQfoo: !llvm.ptr
    %4 = llvm.getelementptr %3[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.call @bar(%4) : (!llvm.ptr) -> ()
    omp.terminator
  }

  llvm.return
}

omp.private {type = private} @_QFlower_region_with_addressof_privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.addressof @_QQfoo: !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}

llvm.mlir.global linkonce constant @_QQfoo() {addr_space = 0 : i32} : !llvm.array<3 x i8> {
  %0 = llvm.mlir.constant("foo") : !llvm.array<3 x i8>
  llvm.return %0 : !llvm.array<3 x i8>
}

llvm.func @bar(!llvm.ptr)


// Tests fix for Fujitsu test suite test: 0275_0032.f90. The MLIR to LLVM
// translation logic assumed that reduction arguments to an `omp.parallel`
// op are always the last set of arguments to the op. However, this is a
// wrong assumption since private args come afterward. This tests the fix
// that we access the different sets of args properly.

// CHECK-LABEL: define internal void @private_and_reduction_..omp_par
// CHECK-DAG:    %[[PRV_ALLOC:.*]] = alloca float, i64 1, align 4
// CHECK-DAG:     %[[RED_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, i64 1, align 8

// CHECK:         omp.par.region:
// CHECK:           br label %[[PAR_REG_BEG:.*]]
// CHECK:         [[PAR_REG_BEG]]:
// CHECK-NEXT:      %{{.*}} = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[RED_ALLOC]], align 8
// CHECK-NEXT:      store float 8.000000e+00, ptr %[[PRV_ALLOC]], align 4

llvm.func @private_and_reduction_() attributes {fir.internal_name = "_QPprivate_and_reduction", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x f32 {bindc_name = "to_priv"} : (i64) -> !llvm.ptr
  omp.parallel private(@privatizer.part %2 -> %arg1 : !llvm.ptr) reduction(byref @reducer.part %1 -> %arg0 : !llvm.ptr) {
    %3 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %4 = llvm.mlir.constant(8.000000e+00 : f32) : f32
    llvm.store %4, %arg1 : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

omp.private {type = private} @privatizer.part : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "to_priv", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}

omp.declare_reduction @reducer.part : !llvm.ptr alloc {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
} init {
^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
  omp.yield(%alloc : !llvm.ptr)
} combiner {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}  cleanup {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

// -----

llvm.func @_QPequivalence() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.array<4 x i8> : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(0 : index) : i64
  %3 = llvm.getelementptr %1[0, %2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  omp.parallel private(@_QFequivalenceEx_firstprivate_ptr_f32 %3 -> %arg0 : !llvm.ptr) {
    %4 = llvm.mlir.constant(3.140000e+00 : f32) : f32
    llvm.store %4, %arg0 : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

omp.private {type = firstprivate} @_QFequivalenceEx_firstprivate_ptr_f32 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "x", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %0, %arg1 : f32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

// CHECK: define internal void @_QPequivalence..omp_par
// CHECK-NOT: define {{.*}} @{{.*}}
// CHECK:   %[[PRIV_ALLOC:.*]] = alloca float, i64 1, align 4
// CHECK:   %[[HOST_VAL:.*]] = load float, ptr %{{.*}}, align 4
// Test that we initialize the firstprivate variable.
// CHECK:   store float %[[HOST_VAL]], ptr %[[PRIV_ALLOC]], align 4
// Test that we inlined the body of the parallel region.
// CHECK:   store float 0x{{.*}}, ptr %[[PRIV_ALLOC]], align 4
