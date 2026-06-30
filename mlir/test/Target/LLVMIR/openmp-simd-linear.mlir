// Ensure that omp.simd with the linear clause is translated correctly even
// when other loop nests exist in the same function.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: void @test_simd_linear()
// CHECK-NOT:     %.linear_result
// CHECK:         %.linear_result = alloca i32

omp.private {type = private} @test_simd_linear_private_i32 : i32
llvm.func @test_simd_linear() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.constant(10 : i32) : i32
  %4 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.parallel {
    omp.wsloop private(@test_simd_linear_private_i32 %5 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%1) to (%3) inclusive step (%1) {
        llvm.store %arg1, %arg0 : i32, !llvm.ptr
        llvm.store %2, %4 : i32, !llvm.ptr
        omp.simd linear(%4 : !llvm.ptr = %1 : i32) {
          omp.loop_nest (%arg2) : i32 = (%1) to (%3) inclusive step (%1) {
            llvm.store %arg2, %4 : i32, !llvm.ptr
            omp.yield
          }
        } {linear_var_types = [i32]}
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}


// -----

// CHECK-LABEL: @test_simd_linear2({{.*}})
// CHECK:       omp.loop_nest.region:
// CHECK:         store i32 %{{.*}}, ptr %.linear_result
// CHECK:         %{{.*}} = load i32, ptr %.linear_result
// CHECK:       omp.region.cont1:

llvm.func @test_simd_linear2(%a : !llvm.ptr) {
  %c0_i64 = llvm.mlir.constant(0 : index) : i64
  %c1_i64 = llvm.mlir.constant(1 : index) : i64
  %c100_i64 = llvm.mlir.constant(100 : index) : i64
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c100_i32 = llvm.mlir.constant(100 : i32) : i32

  %i_ptr = llvm.alloca %c0_i64 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  llvm.br ^bb1(%c1_i32, %c100_i64 : i32, i64)
^bb1(%i: i32, %iv: i64):  // 2 preds: ^bb0, ^bb2
  %1 = llvm.icmp "sgt" %iv, %c0_i64 : i64
  llvm.cond_br %1, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  llvm.store %i, %i_ptr : i32, !llvm.ptr
  %2 = llvm.load %i_ptr : !llvm.ptr -> i32
  %i_next = llvm.add %2, %c1_i32 overflow<nsw> : i32
  %iv_next = llvm.sub %iv, %c1_i64 : i64
  llvm.br ^bb1(%i_next, %iv_next : i32, i64)
^bb3:  // pred: ^bb1
  llvm.store %i, %i_ptr : i32, !llvm.ptr
  omp.simd linear(%i_ptr : !llvm.ptr = %c1_i32 : i32) {
    omp.loop_nest (%arg0) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32) {
      llvm.store %arg0, %i_ptr : i32, !llvm.ptr
      %i2 = llvm.load %i_ptr : !llvm.ptr -> i32
      %3 = llvm.sext %i2 : i32 to i64
      %4 = llvm.sub %3, %c1_i64 overflow<nsw, nuw> : i64
      %5 = llvm.mul %4, %c1_i64 overflow<nsw, nuw> : i64
      %6 = llvm.mul %5, %c1_i64 overflow<nsw, nuw> : i64
      %7 = llvm.add %6, %c0_i64 overflow<nsw, nuw> : i64
      %8 = llvm.getelementptr nusw|nuw %a[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %i2, %8 : i32, !llvm.ptr
      omp.yield
    }
  } {linear_var_types = [i32]}
  llvm.return
}
