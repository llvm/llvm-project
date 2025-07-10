// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Decoding the attribute does not work on big-endian platforms currently
// XFAIL: target={{(s390x|sparc.*)-.*}}

// CHECK{LITERAL}: @dense_resource_tensor_constant = internal constant [5 x float] [float 0x3FCA034080000000, float 0xBFD0466300000000, float 0xBFD75DDF80000000, float 0xBFDE074F40000000, float 0x3FDDD3A1C0000000]
llvm.mlir.global internal constant @dense_resource_tensor_constant(dense_resource<dense_resource_test_5xf32> : tensor<5xf32>) : !llvm.array<5 x f32>

// CHECK{LITERAL}: @dense_resource_vector_constant = internal constant <5 x float> <float 0x3FCA034080000000, float 0xBFD0466300000000, float 0xBFD75DDF80000000, float 0xBFDE074F40000000, float 0x3FDDD3A1C0000000>
llvm.mlir.global internal constant @dense_resource_vector_constant(dense_resource<dense_resource_test_5xf32> : vector<5xf32>) : vector<5xf32>


// CHECK{LITERAL}: @dense_resource_multidim_tensor_constant = internal constant [1 x [2 x [2 x float]]] [[2 x [2 x float]] [[2 x float] [float 0x3FD6B46A80000000, float 0x3FD6781AC0000000], [2 x float] [float 0xBFB45A2AA0000000, float 0x3FD77A5CA0000000]]]
llvm.mlir.global internal constant @dense_resource_multidim_tensor_constant(dense_resource<dense_resource_test_2x2xf32> : tensor<1x2x2xf32>) : !llvm.array<1 x !llvm.array<2 x !llvm.array<2 x f32>>>

// CHECK{LITERAL}: @dense_resource_multidim_vector_constant = internal constant [1 x [2 x <2 x float>]] [[2 x <2 x float>] [<2 x float> <float 0x3FD6B46A80000000, float 0x3FD6781AC0000000>, <2 x float> <float 0xBFB45A2AA0000000, float 0x3FD77A5CA0000000>]]
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
