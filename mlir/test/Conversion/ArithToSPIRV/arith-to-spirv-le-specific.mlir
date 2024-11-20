// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s | FileCheck %s


//===----------------------------------------------------------------------===//
// arith.constant dense_resource
//
// The decoding of dense_resource differs between little and big endian
// machines. At the moment only litte endian is supported.
// See https://github.com/llvm/llvm-project/issues/63469 for more infos.
//
//===----------------------------------------------------------------------===//

// XFAIL: target={{(s390x|sparc.*)-.*}}

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {
func.func @constant_dense_resource() {
  // CHECK:    %{{.*}} = spirv.Constant dense<[0.203224242, -0.254296064, -0.365104556, -0.469196141, 0.466041982]> : tensor<5xf32> : !spirv.array<5 x f32>
  %0 = arith.constant dense_resource<dense_resource_test_5xf32> : tensor<5xf32>  
  // CHECK:    %{{.*}} = spirv.Constant dense<[1, 2]> : vector<2xi32>
  %1 = arith.constant dense_resource<dense_resource_test_2xi32> : vector<2xi32>  
  // CHECK:    %{{.*}} = spirv.Constant dense<[0.35476172, 0.351080596, -0.0795008316, 0.366843373]> : tensor<4xf32> : !spirv.array<4 x f32>
  %2 = arith.constant dense_resource<dense_resource_test_2x2xf32> : tensor<1x2x2xf32>  
  return
  }
}
// Resources are kept at end of file. New tests should be added above this.
{-#
  dialect_resources: {
    builtin: {
      dense_resource_test_2xi32: "0x400000000100000002000000",
      dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E",
      dense_resource_test_2x2xf32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
    }
  }
#-}
