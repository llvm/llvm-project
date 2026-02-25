// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that linear variables in SIMD loops with ordered regions
// are correctly rewritten to use .linear_result in:
// 1. The ordered region (omp.ordered.region)
// 2. Code after the ordered region (omp_region.finalize)
//
// This tests "omp ordered simd" nested in  "omp simd ordered"
// !$omp simd
// do i = 1, n
//     a(i) = b(i) * 10
//         !$omp ordered simd
//             print *, a(i)
//         !$omp end ordered
//     c(i) = a(i) * 2
// end do
// !$omp end simd

module {
  omp.private {type = private} @i_private_i32 : i32

  // CHECK-LABEL: define void @simd_ordered_linear
  llvm.func @simd_ordered_linear() {
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %c10_i32 = llvm.mlir.constant(10 : i32) : i32
    %c10_val = llvm.mlir.constant(10 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32

    // Allocate arrays and loop variable
    %c100_i64 = llvm.mlir.constant(100 : i64) : i64
    %a = llvm.alloca %c100_i64 x i32 : (i64) -> !llvm.ptr
    %b = llvm.alloca %c100_i64 x i32 : (i64) -> !llvm.ptr
    %c = llvm.alloca %c100_i64 x i32 : (i64) -> !llvm.ptr
    %i = llvm.alloca %c1_i64 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr

    // CHECK: %.linear_var = alloca i32
    // CHECK: %.linear_result = alloca i32

    omp.simd linear(%i = %c1_i32 : !llvm.ptr) private(@i_private_i32 %i -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%iv) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
        // CHECK: omp.loop_nest.region:
        // CHECK: load i32, ptr %.linear_result
        llvm.store %iv, %arg0 : i32, !llvm.ptr

        // Compute a[i] = b[i] * 10
        %i_val = llvm.load %arg0 : !llvm.ptr -> i32
        %i_idx = llvm.sext %i_val : i32 to i64
        %i_off = llvm.sub %i_idx, %c1_i64 : i64
        %b_ptr = llvm.getelementptr %b[%i_off] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %b_val = llvm.load %b_ptr : !llvm.ptr -> i32
        %a_val = llvm.mul %b_val, %c10_val : i32
        %a_ptr = llvm.getelementptr %a[%i_off] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %a_val, %a_ptr : i32, !llvm.ptr

        // Ordered region
        omp.ordered.region par_level_simd {
          // CHECK: omp.ordered.region:
          // CHECK: load i32, ptr %.linear_result
          %i_ord = llvm.load %arg0 : !llvm.ptr -> i32
          %i_ord_idx = llvm.sext %i_ord : i32 to i64
          %i_ord_off = llvm.sub %i_ord_idx, %c1_i64 : i64
          %a_ord_ptr = llvm.getelementptr %a[%i_ord_off] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          %a_ord_val = llvm.load %a_ord_ptr : !llvm.ptr -> i32
          omp.terminator
        }

        // Compute c[i] = a[i] * 2 (code after ordered region)
        // CHECK: omp_region.finalize:
        // CHECK: load i32, ptr %.linear_result
        %i_post = llvm.load %arg0 : !llvm.ptr -> i32
        %i_post_idx = llvm.sext %i_post : i32 to i64
        %i_post_off = llvm.sub %i_post_idx, %c1_i64 : i64
        %a_post_ptr = llvm.getelementptr %a[%i_post_off] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %a_post_val = llvm.load %a_post_ptr : !llvm.ptr -> i32
        %c_val = llvm.mul %a_post_val, %c2 : i32
        %c_ptr = llvm.getelementptr %c[%i_post_off] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %c_val, %c_ptr : i32, !llvm.ptr

        omp.yield
      }
    } {linear_var_types = [i32]}
    llvm.return
  }
  // CHECK: !{!"llvm.loop.vectorize.enable", i1 true}
}
