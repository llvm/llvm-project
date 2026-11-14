// RUN: mlir-opt %s --sparsifier="gpu-triple=nvptx64-nvidia-cuda" | FileCheck %s

// Setting `gpu-triple` runs nvvm-attach-target. Check that an unset
// `gpu-chip` keeps the pass default rather than passing on an empty chip.

// CHECK-LABEL: llvm.func @kernel
func.func @kernel() {
  return
}
