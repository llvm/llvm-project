// RUN: mlir-opt %s -arith-expand -split-input-file -verify-diagnostics | FileCheck %s

func.func @scaling_truncf_f32_to_f4E2M1FN(%arg0 : f32, %arg1: f8E8M0FNU) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f32, f8E8M0FNU to f4E2M1FN
    return %0 : f4E2M1FN
}

// CHECK-LABEL: @scaling_truncf_f32_to_f4E2M1FN
// CHECK: %[[SCALEF32:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF32]] : f32
// CHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : f32 to f4E2M1FN
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_truncf_vector_f16_to_f6E3M2FN(%arg0 : vector<4xf16>, %arg1: vector<4xf8E8M0FNU>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf8E8M0FNU> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}

// CHECK-LABEL: @scaling_truncf_vector_f16_to_f6E3M2FN
// CHECK: %[[SCALEF16:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// CHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF16]] : vector<4xf16>
// CHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : vector<4xf16> to vector<4xf6E3M2FN>
// CHECK: return %[[RESULT]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_propagate_rounding_mode_fast_math(%arg0 : vector<4xf16>, %arg1: vector<4xf16>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 to_nearest_even fastmath<fast> : vector<4xf16>, vector<4xf16> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}
// CHECK-LABEL: @scaling_truncf_propagate_rounding_mode_fast_math
// CHECK: %[[SCALEF8:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// CHECK: %[[SCALEINTY:.+]] = arith.extf %[[SCALEF8]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf16>
// CHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEINTY]] fastmath<fast> : vector<4xf16>
// CHECK: %[[TRUNCF:.+]] = arith.truncf [[_:%[a-zA-Z0-9_]+]] to_nearest_even fastmath<fast> : vector<4xf16> to vector<4xf6E3M2FN>
// CHECK: return %[[TRUNCF]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales(%arg0: f16, %arg1 : f16) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f16, f16 to f4E2M1FN
    return %0 : f4E2M1FN
}
// CHECK-LABEL: @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales
// CHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : f16 to f8E8M0FN
// CHECK: return

// -----
func.func @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales(%arg0: vector<4xf16>, %arg1 : vector<4xf16>) -> vector<4xf4E2M1FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf16> to vector<4xf4E2M1FN>
    return %0 : vector<4xf4E2M1FN>
}
// CHECK-LABEL: @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales
// CHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// CHECK: return

// -----

func.func @scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E8M0FNU) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E8M0FNU to f32
    return %0 : f32 
}

// CHECK-LABEL: @scaling_extf_to_f32
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_to_f32_using_f16_scales(%arg0: f4E2M1FN, %arg1 : f16) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f16 to f32
    return %0 : f32 
}

// CHECK-LABEL: @scaling_extf_to_f32_using_f16_scales
// CHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : f16 to f8E8M0FNU
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : f8E8M0FNU to f32
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// CHECK: return %[[RESULT]]

// -----

func.func @invalid_scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E5M2FNUZ) -> f32 {
    // expected-error@+1 {{failed to legalize operation 'arith.scaling_extf' that was explicitly marked illegal}}
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E5M2FNUZ to f32
    return %0 : f32
}

// -----

func.func @scaling_extf_vector_to_f32(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @scaling_extf_vector_to_f32
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf32>
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32> 
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf16>
    return %0 : vector<4xf16>
}

// CHECK-LABEL: @scaling_extf_vector_to_f16
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf16>
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf16> 
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_bf16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xbf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @scaling_extf_vector_to_bf16
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xbf16>
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xbf16>
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xbf16> 
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales
// CHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : vector<4xf8E8M0FNU> to vector<4xf32>
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales_fastmath(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 fastmath<fast> : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales_fastmath
// CHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// CHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf32>
// CHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 fastmath<fast> : vector<4xf4E2M1FN> to vector<4xf32>
// CHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] fastmath<fast> : vector<4xf32>
// CHECK: return %[[RESULT]]
