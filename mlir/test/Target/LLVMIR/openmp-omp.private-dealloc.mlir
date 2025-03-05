// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @free(!llvm.ptr)

llvm.func @parallel_op_dealloc(%arg0: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

omp.private {type = firstprivate} @x.privatizer : f32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %0, %arg1 : f32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.icmp "ne" %0, %c0 : i64
  llvm.cond_br %1, ^bb1, ^bb2

^bb1:
  llvm.call @free(%arg0) : (!llvm.ptr) -> ()
  llvm.br ^bb2

^bb2:
  omp.yield
}

// CHECK-LABEL: define internal void @parallel_op_dealloc..omp_par
// CHECK:         %[[LOCAL_ALLOC:.*]] = alloca float, align 4

// CHECK:      omp.par.pre_finalize:
// CHECK:        br label %[[DEALLOC_REG_START:.*]]

// CHECK:      [[DEALLOC_REG_START]]:
// CHECK:        %[[LOCAL_ALLOC_CONV:.*]] = ptrtoint ptr %[[LOCAL_ALLOC]] to i64
// CHECK:        %[[COND:.*]] = icmp ne i64 %[[LOCAL_ALLOC_CONV]], 0
// CHECK:        br i1 %[[COND]], label %[[DEALLOC_REG_BB1:.*]], label %[[DEALLOC_REG_BB2:.*]]

// CHECK:      [[DEALLOC_REG_BB2]]:

// CHECK:      [[DEALLOC_REG_BB1]]:
// CHECK-NEXT:   call void @free(ptr %[[LOCAL_ALLOC]])
// CHECK-NEXT:   br label %[[DEALLOC_REG_BB2]]
