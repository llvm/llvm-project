// RUN: echo "" | mlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-SAME: acc
// CHECK-SAME: affine
// CHECK-SAME: amdgpu
// CHECK-SAME: amx
// CHECK-SAME: arith
// CHECK-SAME: arm_neon
// CHECK-SAME: arm_sve
// CHECK-SAME: async
// CHECK-SAME: bufferization
// CHECK-SAME: builtin
// CHECK-SAME: cf
// CHECK-SAME: complex
// CHECK-SAME: dlti
// CHECK-SAME: emitc
// CHECK-SAME: func
// CHECK-SAME: gpu
// CHECK-SAME: index
// CHECK-SAME: irdl
// CHECK-SAME: linalg
// CHECK-SAME: llvm
// CHECK-SAME: math
// CHECK-SAME: memref
// CHECK-SAME: ml_program
// CHECK-SAME: nvgpu
// CHECK-SAME: nvvm
// CHECK-SAME: omp
// CHECK-SAME: pdl
// CHECK-SAME: pdl_interp
// CHECK-SAME: quant
// CHECK-SAME: rocdl
// CHECK-SAME: scf
// CHECK-SAME: shape
// CHECK-SAME: sparse_tensor
// CHECK-SAME: spirv
// CHECK-SAME: tensor
// CHECK-SAME: test
// CHECK-SAME: test_dyn
// CHECK-SAME: tosa
// CHECK-SAME: transform
// CHECK-SAME: vector
// CHECK-SAME: x86vector

// RUN: mlir-opt --help-hidden | FileCheck %s -check-prefix=CHECK-HELP
// CHECK-HELP: -p - Alias for --pass-pipeline
