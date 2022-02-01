// RUN: mlir-opt %s -convert-linalg-to-loops -convert-vector-to-scf='full-unroll=true' -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm  -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -convert-arith-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-translate --mlir-to-llvmir |\
// RUN: %lli --entry-function=entry --mattr="avx512f" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext |\
// RUN: FileCheck %s

module {

  // an array of 16 i32 of values [0..15]
  llvm.mlir.global private @const16(
    dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16 x i32>)
      : !llvm.array<16 x i32>

  llvm.func @entry() -> i32 {
    %c0 = llvm.mlir.constant(0 : index) : i64

    %1 = llvm.mlir.addressof @const16 : !llvm.ptr<array<16 x i32>>
    %ptr = llvm.getelementptr %1[%c0, %c0]
      : (!llvm.ptr<array<16 x i32>>, i64, i64) -> !llvm.ptr<i32>
    %ptr2 = llvm.bitcast %ptr :  !llvm.ptr<i32> to !llvm.ptr<vector<16xi32>>

    // operand_attrs of *m operands need to be piped through to LLVM for
    // verification to pass.
    %v = llvm.inline_asm
        asm_dialect = intel
        operand_attrs = [{ elementtype = vector<16xi32> }]
        "vmovdqu32 $0, $1", "=x,*m" %ptr2
      : (!llvm.ptr<vector<16xi32>>) -> vector<16xi32>

    // CHECK: 0
    %v0 = vector.extract %v[0]: vector<16xi32>
    vector.print %v0 : i32

    // CHECK: 9
    %v9 = vector.extract %v[9]: vector<16xi32>
    vector.print %v9 : i32

    %i0 = arith.constant 0 : i32
    llvm.return %i0 : i32
  }
}

