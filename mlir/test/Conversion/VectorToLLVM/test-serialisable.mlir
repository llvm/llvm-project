// Simple regression test that ensures ConvertVectorToLLVMPass options remain
// serialisable. We don't need to actually parse any IR to print the pass
// options. We just need to provide --dump-pass-pipeline

// RUN: mlir-opt --convert-vector-to-llvm --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// RUN: mlir-opt --convert-vector-to-llvm='vector-contract-lowering=matmul vector-transpose-lowering=flat' \
// RUN:          --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefix=CHANGED

// CHECK: builtin.module(
// CHECK-SAME: convert-vector-to-llvm{
// CHECK-SAME: enable-amx={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-arm-neon={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-arm-sve={{[aA-zZ0-9]+}}
// CHECK-SAME: enable-x86vector={{[aA-zZ0-9]+}}
// CHECK-SAME: force-32bit-vector-indices={{[aA-zZ0-9]+}}
// CHECK-SAME: reassociate-fp-reductions={{[aA-zZ0-9]+}}
// DEFAULT: vector-contract-lowering=dot
// DEFAULT: vector-transpose-lowering=eltwise
// CHANGED: vector-contract-lowering=matmul
// CHANGED: vector-transpose-lowering=flat
