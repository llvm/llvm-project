// REQUIRES: target=x86{{.*}}

// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-llvm="enable-x86vector" -convert-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-translate --mlir-to-llvmir | \
// RUN: llc -mcpu=sierraforest | \
// RUN: FileCheck %s

func.func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_128(%arg0: memref<8xbf16>) -> vector<4xf32> {
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8xbf16> -> index
  %0 = arith.index_cast %intptr : index to i32
  %1 = llvm.inttoptr %0 : i32 to !llvm.ptr
  %2 = x86vector.avx.cvt.packed.even.indexed.bf16_to_f32 %1 : !llvm.ptr -> vector<4xf32>
  return %2 : vector<4xf32>
}
// CHECK-LABEL: avxbf16_cvt_packed_even_indexed_bf16_to_f32_128:
// CHECK: vcvtneebf162ps{{.*}}%xmm

func.func @avxbf16_cvt_packed_even_indexed_bf16_to_f32_256(%arg0: memref<16xbf16>) -> vector<8xf32> {
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<16xbf16> -> index
  %0 = arith.index_cast %intptr : index to i32
  %1 = llvm.inttoptr %0 : i32 to !llvm.ptr
  %2 = x86vector.avx.cvt.packed.even.indexed.bf16_to_f32 %1 : !llvm.ptr -> vector<8xf32>
  return %2 : vector<8xf32>
}
// CHECK-LABEL: avxbf16_cvt_packed_even_indexed_bf16_to_f32_256:
// CHECK: vcvtneebf162ps{{.*}}%ymm

func.func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_128(%arg0: memref<8xbf16>) -> vector<4xf32> {
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8xbf16> -> index
  %0 = arith.index_cast %intptr : index to i32
  %1 = llvm.inttoptr %0 : i32 to !llvm.ptr
  %2 = x86vector.avx.cvt.packed.odd.indexed.bf16_to_f32 %1 : !llvm.ptr -> vector<4xf32>
  return %2 : vector<4xf32>
}
// CHECK-LABEL: avxbf16_cvt_packed_odd_indexed_bf16_to_f32_128:
// CHECK: vcvtneobf162ps{{.*}}%xmm

func.func @avxbf16_cvt_packed_odd_indexed_bf16_to_f32_256(%arg0: memref<16xbf16>) -> vector<8xf32> {
  %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<16xbf16> -> index
  %0 = arith.index_cast %intptr : index to i32
  %1 = llvm.inttoptr %0 : i32 to !llvm.ptr
  %2 = x86vector.avx.cvt.packed.odd.indexed.bf16_to_f32 %1 : !llvm.ptr -> vector<8xf32>
  return %2 : vector<8xf32>
}
// CHECK-LABEL: avxbf16_cvt_packed_odd_indexed_bf16_to_f32_256:
// CHECK: vcvtneobf162ps{{.*}}%ymm
