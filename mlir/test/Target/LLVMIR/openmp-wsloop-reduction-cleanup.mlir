// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// test a wsloop reduction with a cleanup region

  omp.declare_reduction @add_reduction_i_32 : !llvm.ptr init {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.mlir.constant(0 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.call @malloc(%c4) : (i64) -> !llvm.ptr
    llvm.store %0, %2 : i32, !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    %1 = llvm.load %arg1 : !llvm.ptr -> i32
    %2 = llvm.add %0, %1  : i32
    llvm.store %2, %arg0 : i32, !llvm.ptr
    omp.yield(%arg0 : !llvm.ptr)
  } cleanup {
  ^bb0(%arg0: !llvm.ptr):
    llvm.call @free(%arg0) : (!llvm.ptr) -> ()
    omp.yield
  }

  // CHECK-LABEL: @main
  llvm.func @main()  {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.addressof @i : !llvm.ptr
    %2 = llvm.mlir.addressof @j : !llvm.ptr
    %loop_ub = llvm.mlir.constant(9 : i32) : i32
    %loop_lb = llvm.mlir.constant(0 : i32) : i32
    %loop_step = llvm.mlir.constant(1 : i32) : i32 
    omp.wsloop reduction(byref @add_reduction_i_32 %1 -> %arg0 : !llvm.ptr, byref @add_reduction_i_32 %2 -> %arg1 : !llvm.ptr) {
      omp.loop_nest (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
        llvm.store %0, %arg0 : i32, !llvm.ptr
        llvm.store %0, %arg1 : i32, !llvm.ptr
        omp.yield
      }
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @i() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.mlir.global internal @j() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @malloc(%arg0 : i64) -> !llvm.ptr
  llvm.func @free(%arg0 : !llvm.ptr) -> ()

// Private reduction variable and its initialization.
// CHECK: %[[MALLOC_I:.+]] = call ptr @malloc(i64 4)
// CHECK: %[[PRIV_PTR_I:.+]] = alloca ptr
// CHECK: store ptr %[[MALLOC_I]], ptr %[[PRIV_PTR_I]]
// CHECK: %[[MALLOC_J:.+]] = call ptr @malloc(i64 4)
// CHECK: %[[PRIV_PTR_J:.+]] = alloca ptr
// CHECK: store ptr %[[MALLOC_J]], ptr %[[PRIV_PTR_J]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Weirdly the finalization block is generated before the reduction blocks:
// CHECK: [[FINALIZE:.+]]:
// CHECK: call void @__kmpc_barrier
// CHECK: %[[PRIV_I:.+]] = load ptr, ptr %[[PRIV_PTR_I]], align 8
// CHECK: call void @free(ptr %[[PRIV_I]])
// CHECK: %[[PRIV_J:.+]] = load ptr, ptr %[[PRIV_PTR_J]], align 8
// CHECK: call void @free(ptr %[[PRIV_J]])
// CHECK: ret void

// Non-atomic reduction:
// CHECK: %[[PRIV_VAL_PTR_I:.+]] = load ptr, ptr %[[PRIV_PTR_I]]
// CHECK: %[[LOAD_I:.+]] = load i32, ptr @i
// CHECK: %[[PRIV_VAL_I:.+]] = load i32, ptr %[[PRIV_VAL_PTR_I]]
// CHECK: %[[SUM_I:.+]] = add i32 %[[LOAD_I]], %[[PRIV_VAL_I]]
// CHECK: store i32 %[[SUM_I]], ptr @i
// CHECK: %[[PRIV_VAL_PTR_J:.+]] = load ptr, ptr %[[PRIV_PTR_J]]
// CHECK: %[[LOAD_J:.+]] = load i32, ptr @j
// CHECK: %[[PRIV_VAL_J:.+]] = load i32, ptr %[[PRIV_VAL_PTR_J]]
// CHECK: %[[SUM_J:.+]] = add i32 %[[LOAD_J]], %[[PRIV_VAL_J]]
// CHECK: store i32 %[[SUM_J]], ptr @j
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: add i32
