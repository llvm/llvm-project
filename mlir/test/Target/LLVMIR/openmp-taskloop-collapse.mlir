// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFtestEi_private_i32 : i32

omp.private {type = firstprivate} @_QFtestEa_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}


llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3) : i32 = (%c1_i32, %c1_i32) to (%c10_i32, %c5_i32) inclusive step (%c1_i32, %c1_i32) collapse(2) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 50, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %gep_ub.val, align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_6:.*]] = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 5
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 5
// CHECK: %[[VAL_11:.*]] = mul i32 %[[VAL_10]], 1
// CHECK: %[[VAL_12:.*]] = add i32 %[[VAL_11]], 1
// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_9]], 1
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 1

// -----

llvm.func @_QPtest2() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3, %arg4) : i32 = (%c1_i32, %c1_i32, %c2_i32) to (%c10_i32, %c5_i32, %c5_i32) inclusive step (%c1_i32, %c1_i32, %c1_i32) collapse(3) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 200, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %gep_ub.val, align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_6:.*]] = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 4
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 4
// CHECK: %[[VAL_11:.*]] = urem i32 %[[VAL_10]], 5
// CHECK: %[[VAL_12:.*]] = udiv i32 %[[VAL_10]], 5
// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_12]], 1
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 1
// CHECK: %[[VAL_15:.*]] = mul i32 %[[VAL_11]], 1
// CHECK: %[[VAL_16:.*]] = add i32 %[[VAL_15]], 1
// CHECK: %[[VAL_17:.*]] = mul i32 %[[VAL_9]], 1
// CHECK: %[[VAL_18:.*]] = add i32 %[[VAL_17]], 2

// -----

llvm.func @_QPtest3() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  %c20_i32 = llvm.mlir.constant(20 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3) : i32 = (%c10_i32, %c1_i32) to (%c20_i32, %c5_i32) inclusive step (%c1_i32, %c1_i32) collapse(2) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 55, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %[[gep_task_ub]], align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_5:.*]] = trunc i64 %[[trip_cnt]] to i32
// CHECK: %6 = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 5
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 5

// CHECK: %[[VAL_11:.*]] = mul i32 %[[VAL_10]], 1
// CHECK: %[[VAL_12:.*]] = add i32 %[[VAL_11]], 10

// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_9]], 1
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 1

// -----

llvm.func @_QPtest4() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  %c15_i32 = llvm.mlir.constant(15 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3) : i32 = (%c2_i32, %c5_i32) to (%c10_i32, %c15_i32) inclusive step (%c2_i32, %c3_i32) collapse(2) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 20, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %[[gep_task_ub]], align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_5:.*]] = trunc i64 %[[trip_cnt]] to i32
// CHECK: %6 = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 4
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 4

// CHECK: %[[VAL_11:.*]] = mul i32 %[[VAL_10]], 2
// CHECK: %[[VAL_12:.*]] = add i32 %[[VAL_11]], 2

// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_9]], 3
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 5


// -----

llvm.func @_QPtest5() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %cneg2_i32 = llvm.mlir.constant(-2: i32) : i32
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  %c3_i32 = llvm.mlir.constant(3 : i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  %c15_i32 = llvm.mlir.constant(15 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3) : i32 = (%cneg2_i32, %c5_i32) to (%c10_i32, %c15_i32) inclusive step (%c2_i32, %c3_i32) collapse(2) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 28, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %[[gep_task_ub]], align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_5:.*]] = trunc i64 %[[trip_cnt]] to i32
// CHECK: %6 = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 4
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 4

// CHECK: %[[VAL_11:.*]] = mul i32 %[[VAL_10]], 2
// CHECK: %[[VAL_12:.*]] = add i32 %[[VAL_11]], -2

// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_9]], 3
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 5

// -----

llvm.func @_QPtest6() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %cneg1_i32 = llvm.mlir.constant(-1: i32) : i32
  %c1_i32 = llvm.mlir.constant(1 :i32) : i32
  %c5_i32 = llvm.mlir.constant(5 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  omp.taskloop private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2, %arg3) : i32 = (%c10_i32, %c1_i32) to (%c5_i32, %c5_i32) inclusive step (%cneg1_i32, %c1_i32) collapse(2) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK: %[[structArg:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK: %[[ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[structArg]], i32 0, i32 1
// CHECK: store i64 30, ptr %[[ub]], align 4

// CHECK: %[[VAL_1:.*]] = load ptr, ptr %0, align 8
// CHECK: %[[gep_task_lb:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 0
// CHECK: %[[task_lb:.*]] = load i64, ptr %[[gep_task_lb]], align 4
// CHECK: %[[gep_task_ub:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[VAL_1]], i32 0, i32 1
// CHECK: %[[task_ub:.*]] = load i64, ptr %[[gep_task_ub]], align 4

// CHECK: %[[VAL_3:.*]] = sub i64 %[[task_ub]], %[[task_lb]]
// CHECK: %[[VAL_4:.*]] = sdiv i64 %[[VAL_3]], 1
// CHECK: %[[trip_cnt:.*]] = add i64 %[[VAL_4]], 1
// CHECK: %[[VAL_5:.*]] = trunc i64 %[[trip_cnt]] to i32
// CHECK: %6 = trunc i64 %[[task_lb]] to i32

// CHECK: %[[VAL_7:.*]] = sub i32 %[[VAL_6]], 1
// CHECK: %[[VAL_8:.*]] = add i32 %omp_collapsed.iv, %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = urem i32 %[[VAL_8]], 5
// CHECK: %[[VAL_10:.*]] = udiv i32 %[[VAL_8]], 5

// CHECK: %[[VAL_11:.*]] = mul i32 %[[VAL_10]], -1
// CHECK: %[[VAL_12:.*]] = add i32 %[[VAL_11]], 10

// CHECK: %[[VAL_13:.*]] = mul i32 %[[VAL_9]], 1
// CHECK: %[[VAL_14:.*]] = add i32 %[[VAL_13]], 1
