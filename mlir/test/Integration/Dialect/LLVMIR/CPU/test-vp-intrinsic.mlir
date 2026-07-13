// RUN: mlir-opt %s -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: -convert-vector-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN: -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry \
// RUN:      --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// %mlir_native_utils_lib_dir is incorrect on Windows
// UNSUPPORTED: system-windows

memref.global "private" @gv_i32 : memref<20xi32> =
    dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
           10, 11, 12, 13, 14, 15, 16, 17, 18, 19]>

func.func @entry() -> i32 {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %mem_i32 = memref.get_global @gv_i32 : memref<20xi32>
  // When the vectors are defined as dense constant vector,
  // the vp intrinsic will be optimized/eliminated on some backend (e.g. X86).
  // So this test case loads the vector from a memref to test the vp intrinsic
  // backend support.
  %vec1 = vector.load %mem_i32[%c0] : memref<20xi32>, vector<8xi32>
  %vec2 = vector.load %mem_i32[%c10] : memref<20xi32>, vector<8xi32>
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 0, 1, 0]> : vector<8xi1>
  %evl = arith.constant 4 : i32

  %res = "llvm.intr.vp.add" (%vec1, %vec2, %mask, %evl) :
         (vector<8xi32>, vector<8xi32>, vector<8xi1>, i32) -> vector<8xi32>
  vector.print %res : vector<8xi32>
  // CHECK: ( 10, {{.*}}, 14, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} )

  %ret = arith.constant 0 : i32
  return %ret : i32
}
