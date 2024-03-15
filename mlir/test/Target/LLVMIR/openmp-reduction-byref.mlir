// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

  omp.reduction.declare @add_reduction_i_32 : !llvm.ptr init {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 : (i64) -> !llvm.ptr
    llvm.store %0, %2 : i32, !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    %1 = llvm.load %arg1 : !llvm.ptr -> i32
    %2 = llvm.add %0, %1  : i32
    llvm.store %2, %arg0 : i32, !llvm.ptr
    omp.yield(%arg0 : !llvm.ptr)
  } 

  // CHECK-LABEL: @main
  llvm.func @main()  {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.addressof @i : !llvm.ptr
    omp.parallel byref reduction(@add_reduction_i_32 %1 -> %arg0 : !llvm.ptr) {
      llvm.store %0, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @i() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

// CHECK: %{{.+}} = 
// Call to the outlined function.
// CHECK: call void {{.*}} @__kmpc_fork_call
// CHECK-SAME: @[[OUTLINED:[A-Za-z_.][A-Za-z0-9_.]*]]

// Outlined function.
// CHECK: define internal void @[[OUTLINED]]

// Private reduction variable and its initialization.
// CHECK: %tid.addr.local = alloca i32
// CHECK: %[[PRIVATE:.+]] = alloca i32
// CHECK: store i32 0, ptr %[[PRIVATE]]
// CHECK: store ptr %[[PRIVATE]], ptr %[[PRIV_PTR:.+]],

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]


// Non-atomic reduction:
// CHECK: %[[PRIV_VAL_PTR:.+]] = load ptr, ptr %[[PRIV_PTR]]
// CHECK: %[[LOAD:.+]] = load i32, ptr @i
// CHECK: %[[PRIV_VAL:.+]] = load i32, ptr %[[PRIV_VAL_PTR]]
// CHECK: %[[SUM:.+]] = add i32 %[[LOAD]], %[[PRIV_VAL]]
// CHECK: store i32 %[[SUM]], ptr @i
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: add i32
