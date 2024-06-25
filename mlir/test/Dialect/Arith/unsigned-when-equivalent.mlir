// RUN: mlir-opt -arith-unsigned-when-equivalent %s | FileCheck %s

// CHECK-LABEL: func @not_with_maybe_overflow
// CHECK: arith.divsi
// CHECK: arith.ceildivsi
// CHECK: arith.floordivsi
// CHECK: arith.remsi
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK: arith.extsi
// CHECK: arith.cmpi sle
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi sgt
func.func @not_with_maybe_overflow(%arg0 : i32) {
    %ci32_smax = arith.constant 0x7fffffff : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %0 = arith.minui %arg0, %ci32_smax : i32
    %1 = arith.addi %0, %c1 : i32
    %2 = arith.divsi %1, %c4 : i32
    %3 = arith.ceildivsi %1, %c4 : i32
    %4 = arith.floordivsi %1, %c4 : i32
    %5 = arith.remsi %1, %c4 : i32
    %6 = arith.minsi %1, %c4 : i32
    %7 = arith.maxsi %1, %c4 : i32
    %8 = arith.extsi %1 : i32 to i64
    %9 = arith.cmpi sle, %1, %c4 : i32
    %10 = arith.cmpi slt, %1, %c4 : i32
    %11 = arith.cmpi sge, %1, %c4 : i32
    %12 = arith.cmpi sgt, %1, %c4 : i32
    func.return
}

// CHECK-LABEL: func @yes_with_no_overflow
// CHECK: arith.divui
// CHECK: arith.ceildivui
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.minui
// CHECK: arith.maxui
// CHECK: arith.extui
// CHECK: arith.cmpi ule
// CHECK: arith.cmpi ult
// CHECK: arith.cmpi uge
// CHECK: arith.cmpi ugt
func.func @yes_with_no_overflow(%arg0 : i32) {
    %ci32_almost_smax = arith.constant 0x7ffffffe : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %0 = arith.minui %arg0, %ci32_almost_smax : i32
    %1 = arith.addi %0, %c1 : i32
    %2 = arith.divsi %1, %c4 : i32
    %3 = arith.ceildivsi %1, %c4 : i32
    %4 = arith.floordivsi %1, %c4 : i32
    %5 = arith.remsi %1, %c4 : i32
    %6 = arith.minsi %1, %c4 : i32
    %7 = arith.maxsi %1, %c4 : i32
    %8 = arith.extsi %1 : i32 to i64
    %9 = arith.cmpi sle, %1, %c4 : i32
    %10 = arith.cmpi slt, %1, %c4 : i32
    %11 = arith.cmpi sge, %1, %c4 : i32
    %12 = arith.cmpi sgt, %1, %c4 : i32
    func.return
}

// CHECK-LABEL: func @preserves_structure
// CHECK: scf.for %[[arg1:.*]] =
// CHECK: %[[v:.*]] = arith.remui %[[arg1]]
// CHECK: %[[w:.*]] = arith.addi %[[v]], %[[v]]
// CHECK: %[[test:.*]] = arith.cmpi ule, %[[w]]
// CHECK: scf.if %[[test]]
// CHECK: memref.store %[[w]]
func.func @preserves_structure(%arg0 : memref<8xindex>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %arg1 = %c0 to %c8 step %c1 {
        %v = arith.remsi %arg1, %c4 : index
        %w = arith.addi %v, %v : index
        %test = arith.cmpi sle, %w, %c4 : index
        scf.if %test {
            memref.store %w, %arg0[%arg1] : memref<8xindex>
        }
    }
    func.return
}

func.func private @external() -> i8

// CHECK-LABEL: @dead_code
func.func @dead_code() {
  %0 = call @external() : () -> i8
  // CHECK: arith.floordivsi
  %1 = arith.floordivsi %0, %0 : i8
  return
}

// Make sure not crash.
// CHECK-LABEL: @no_integer_or_index
func.func @no_integer_or_index() { 
  // CHECK: arith.cmpi
  %cst_0 = arith.constant dense<[0]> : vector<1xi32> 
  %cmp = arith.cmpi slt, %cst_0, %cst_0 : vector<1xi32> 
  return
}

// CHECK-LABEL: @gpu_func
func.func @gpu_func(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token, %arg5: index, %arg6: index) -> memref<2x32xf32> {
  %c1 = arith.constant 1 : index  
  %2 = gpu.launch async [%arg4] blocks(%arg7, %arg8, %arg9) in (%arg13 = %c1, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c1, %arg17 = %c1, %arg18 = %c1) {
    gpu.terminator
  } 
  return %arg1 : memref<2x32xf32> 
}  
