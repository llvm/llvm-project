// Bug: https://github.com/llvm/llvm-project/issues/132894

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'func.func' op entry block must have 1 arguments to match function signature

module {
  func.func @f(f32) {
    %c0 = arith.constant 0 : index
    return
  }
}

