// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Decoding the attribute does not work on big-endian platforms currently
// XFAIL: target={{(s390x|sparc.*)-.*}}
// XFAIL: system-aix

// CHECK{LITERAL}: @dense_resource_tensor_constant = internal constant [5 x float] [float f0x3E501A04, float f0xBE823318, float f0xBEBAEEFC, float f0xBEF03A7A, float f0x3EEE9D0E]
llvm.mlir.global internal constant @dense_resource_tensor_constant(dense_resource<dense_resource_test_5xf32> : tensor<5xf32>) : !llvm.array<5 x f32>

// CHECK{LITERAL}: @dense_resource_vector_constant = internal constant <5 x float> <float f0x3E501A04, float f0xBE823318, float f0xBEBAEEFC, float f0xBEF03A7A, float f0x3EEE9D0E>
llvm.mlir.global internal constant @dense_resource_vector_constant(dense_resource<dense_resource_test_5xf32> : vector<5xf32>) : vector<5xf32>


// CHECK{LITERAL}: @dense_resource_multidim_tensor_constant = internal constant [1 x [2 x [2 x float]]] [[2 x [2 x float]] [[2 x float] [float f0x3EB5A354, float f0x3EB3C0D6], [2 x float] [float f0xBDA2D155, float f0x3EBBD2E5]]] 
llvm.mlir.global internal constant @dense_resource_multidim_tensor_constant(dense_resource<dense_resource_test_2x2xf32> : tensor<1x2x2xf32>) : !llvm.array<1 x !llvm.array<2 x !llvm.array<2 x f32>>>

// CHECK{LITERAL}: @dense_resource_multidim_vector_constant = internal constant [1 x [2 x <2 x float>]] [[2 x <2 x float>] [<2 x float> <float f0x3EB5A354, float f0x3EB3C0D6>, <2 x float> <float f0xBDA2D155, float f0x3EBBD2E5>]]
llvm.mlir.global internal constant @dense_resource_multidim_vector_constant(dense_resource<dense_resource_test_2x2xf32> : vector<1x2x2xf32>) : !llvm.array<1 x !llvm.array<2 x vector<2 x f32>>>

// Resources are kept at end of file. New tests should be added above this.
{-#
  dialect_resources: {
    builtin: {
      dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E",
      dense_resource_test_2x2xf32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
    }
  }
#-}
