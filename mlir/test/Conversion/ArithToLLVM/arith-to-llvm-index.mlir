// RUN: mlir-opt %s -convert-arith-to-llvm="index-bitwidth=32" | FileCheck %s --check-prefix=INDEX32
// RUN: mlir-opt %s -convert-arith-to-llvm="index-bitwidth=64" | FileCheck %s --check-prefix=INDEX64

// The sum's type must be converted to the configured index width when building
// the LLVM struct that holds the sum and overflow indicator.

// INDEX32-LABEL: func.func @addui_extended_index(
// INDEX32: "llvm.intr.uadd.with.overflow"{{.*}} : (i32, i32) -> !llvm.struct<(i32, i1)>
// INDEX64-LABEL: func.func @addui_extended_index(
// INDEX64: "llvm.intr.uadd.with.overflow"{{.*}} : (i64, i64) -> !llvm.struct<(i64, i1)>
func.func @addui_extended_index(%a: index, %b: index) -> (index, i1) {
  %sum, %overflow = arith.addui_extended %a, %b : index, i1
  return %sum, %overflow : index, i1
}
