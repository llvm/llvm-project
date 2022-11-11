// RUN: mlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-NEXT: acc
// CHECK-NEXT: affine
// CHECK-NEXT: amdgpu
// CHECK-NEXT: amx
// CHECK-NEXT: arith
// CHECK-NEXT: arm_neon
// CHECK-NEXT: arm_sve
// CHECK-NEXT: async
// CHECK-NEXT: bufferization
// CHECK-NEXT: builtin
// CHECK-NEXT: cf
// CHECK-NEXT: complex
// CHECK-NEXT: dlti
// CHECK-NEXT: emitc
// CHECK-NEXT: func
// CHECK-NEXT: gpu
// CHECK-NEXT: index
// CHECK-NEXT: linalg
// CHECK-NEXT: llvm
// CHECK-NEXT: math
// CHECK-NEXT: memref
// CHECK-NEXT: ml_program
// CHECK-NEXT: nvgpu
// CHECK-NEXT: nvvm
// CHECK-NEXT: omp
// CHECK-NEXT: pdl
// CHECK-NEXT: pdl_interp
// CHECK-NEXT: quant
// CHECK-NEXT: rocdl
// CHECK-NEXT: scf
// CHECK-NEXT: shape
// CHECK-NEXT: sparse_tensor
// CHECK-NEXT: spirv
// CHECK-NEXT: tensor
// CHECK-NEXT: test
// CHECK-NEXT: test_dyn
// CHECK-NEXT: tosa
// CHECK-NEXT: transform
// CHECK-NEXT: vector
// CHECK-NEXT: x86vector

// RUN: mlir-opt --help-hidden | FileCheck %s -check-prefix=CHECK-HELP
// CHECK-HELP: -p - Alias for --pass-pipeline
