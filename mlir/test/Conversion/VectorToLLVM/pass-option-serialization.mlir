// Ensure that ConvertVectorToLLVMPass options remain serialisable.

// This test also allows us to exercise these options (to some extent) even if we
// don't use them in other Vector to LLVM conversion tests. This is quite relevant
// for the `Vector` Dialect (and `--convert-vector-to-llvm` pass) as in many cases
// we use the Transform Dialect (TD) rather than `--convert-vector-to-llvm` for
// testing. So here we don't check the correctness of the passes, as they're
// covered by other tests that use TD, but we still provide some test coverage of
// these pass options.

// We don't need to actually parse any IR to print the pass options. We just need
// to provide --dump-pass-pipeline

// RUN: mlir-opt --convert-vector-to-llvm --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// RUN: mlir-opt --convert-vector-to-llvm='vector-contract-lowering=llvmintr vector-transpose-lowering=llvmintr' \
// RUN:          --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=NON-DEFAULT

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
// NON-DEFAULT: vector-contract-lowering=llvm
// NON-DEFAULT: vector-transpose-lowering=llvm
