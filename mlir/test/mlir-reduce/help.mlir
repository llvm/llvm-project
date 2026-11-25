// RUN: mlir-reduce --help | FileCheck %s

// The tool's own options must be listed by --help. They are hidden
// if they are not in the "mlir-reduce options" category.
// CHECK-LABEL: mlir-reduce options:
// CHECK-DAG: --allow-unregistered-dialect
// CHECK-DAG: --no-implicit-module
// CHECK-DAG: --split-input-file
