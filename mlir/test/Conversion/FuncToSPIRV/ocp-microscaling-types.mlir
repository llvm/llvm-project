// RUN: mlir-opt --split-input-file --convert-func-to-spirv %s | FileCheck %s --check-prefixes=CHECK,EMU
// RUN: mlir-opt --split-input-file --convert-func-to-spirv="emulate-unsupported-float-types=false" %s | FileCheck %s --check-prefixes=CHECK,NATIVE

// Without the microscaling extension, these types cannot be represented by
// widening to f32.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: func.func @unsupported_fp4
  // CHECK-SAME: f4E2M1FN
  func.func @unsupported_fp4(%arg: f4E2M1FN) { return }

  // CHECK-LABEL: func.func @unsupported_fp6e2m3
  // CHECK-SAME: f6E2M3FN
  func.func @unsupported_fp6e2m3(%arg: f6E2M3FN) { return }

  // CHECK-LABEL: func.func @unsupported_fp6e3m2
  // CHECK-SAME: f6E3M2FN
  func.func @unsupported_fp6e3m2(%arg: f6E3M2FN) { return }

  // CHECK-LABEL: func.func @unsupported_fp4_vector
  // CHECK-SAME: vector<2xf4E2M1FN>
  func.func @unsupported_fp4_vector(%arg: vector<2xf4E2M1FN>) { return }

  // CHECK-LABEL: func.func @unsupported_fp6_tensor
  // CHECK-SAME: tensor<2xf6E2M3FN>
  func.func @unsupported_fp6_tensor(%arg: tensor<2xf6E2M3FN>) { return }
}

// -----

// With the required capabilities and extensions, preserve the native types.
module attributes {
  spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.0, [Float4EXT, Float6EXT, Float8UnsignedE8M0EXT],
                 [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
      #spirv.resource_limits<>>
} {
  // CHECK-LABEL: spirv.func @supported_scalars
  // CHECK-SAME: f4E2M1FN
  // CHECK-SAME: f6E2M3FN
  // CHECK-SAME: f6E3M2FN
  // CHECK-SAME: f8E8M0FNU
  func.func @supported_scalars(%f4: f4E2M1FN, %f6e2m3: f6E2M3FN,
                               %f6e3m2: f6E3M2FN, %f8: f8E8M0FNU) {
    return
  }

  // CHECK-LABEL: spirv.func @supported_vectors
  // CHECK-SAME: vector<2xf4E2M1FN>
  // CHECK-SAME: vector<2xf6E2M3FN>
  // CHECK-SAME: vector<2xf6E3M2FN>
  // EMU-SAME: vector<2xi32>
  // NATIVE-SAME: vector<2xf8E8M0FNU>
  func.func @supported_vectors(%f4: vector<2xf4E2M1FN>,
                               %f6e2m3: vector<2xf6E2M3FN>,
                               %f6e3m2: vector<2xf6E3M2FN>,
                               %f8: vector<2xf8E8M0FNU>) {
    return
  }
}

// -----

// Microscaling types cannot be used as image elements, even when the target
// supports the formats. Leave these functions unconverted.
module attributes {
  spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.0, [Float4EXT, Float6EXT, Float8UnsignedE8M0EXT],
                 [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
      #spirv.resource_limits<>>
} {
  // CHECK-LABEL: func.func @unsupported_fp4_image
  // CHECK-SAME: memref<4xf4E2M1FN, #spirv.storage_class<Image>>
  func.func @unsupported_fp4_image(%arg: memref<4xf4E2M1FN, #spirv.storage_class<Image>>) { return }

  // CHECK-LABEL: func.func @unsupported_fp6e2m3_image
  // CHECK-SAME: memref<4xf6E2M3FN, #spirv.storage_class<Image>>
  func.func @unsupported_fp6e2m3_image(%arg: memref<4xf6E2M3FN, #spirv.storage_class<Image>>) { return }

  // CHECK-LABEL: func.func @unsupported_fp6e3m2_image
  // CHECK-SAME: memref<4xf6E3M2FN, #spirv.storage_class<Image>>
  func.func @unsupported_fp6e3m2_image(%arg: memref<4xf6E3M2FN, #spirv.storage_class<Image>>) { return }

  // CHECK-LABEL: func.func @unsupported_e8m0_image
  // CHECK-SAME: memref<4xf8E8M0FNU, #spirv.storage_class<Image>>
  func.func @unsupported_e8m0_image(%arg: memref<4xf8E8M0FNU, #spirv.storage_class<Image>>) { return }
}
