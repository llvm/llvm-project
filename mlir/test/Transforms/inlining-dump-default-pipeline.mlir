// RUN: mlir-opt %s -pass-pipeline="builtin.module(inline)" -dump-pass-pipeline 2>&1 | FileCheck %s
// CHECK: builtin.module(inline{max-iterations=4 pre-inline-pipeline=canonicalize})
