// RUN: mlir-opt %s -pass-pipeline="builtin.module(inline)" -dump-pass-pipeline 2>&1 | FileCheck %s
// CHECK: builtin.module(inline{default-pipeline=canonicalize max-iterations=4 })
