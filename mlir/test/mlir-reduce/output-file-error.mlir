// RUN: not mlir-reduce %s -o %t.missing-dir/out.mlir 2>&1 | FileCheck %s

// mlir-reduce must say why it failed when the output file can't be opened.
// CHECK: cannot open output file '{{.*}}out.mlir'

func.func @f() {
    return
}
