// RUN: mlir-opt --convert-vector-to-llvm --dump-pass-pipeline %s 2>&1 | FileCheck %s

// Simple regression test that ensures ConvertVectorToLLVMPass options remain
// serialisable. We don't need to actually parse any IR to print the pass
// options. We just need to provide --dump-pass-pipeline

// CHECK: builtin.module(
// CHECK-SAME: convert-vector-to-llvm{
// CHECK-SAME: enable-amx={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-arm-neon={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-arm-sve={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-x86vector={{[aA-zZ0-9]+}}
// CHECK-SAME: force-32bit-vector-indices={{[aA-zZ0-9]+}}
// CHECK-SAME: reassociate-fp-reductions={{[aA-zZ0-9]+}}
// CHECK-SAME: vector-contract-lowering={{[aA-zZ0-9]+}}
// CHECK-SAME: vector-transpose-lowering={{[aA-zZ0-9]+}}})
