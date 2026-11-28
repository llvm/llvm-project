// RUN: mlir-opt --split-input-file --verify-diagnostics %s

func.func @same_width(%base: i32, %offset: i32) -> f32 {
  // expected-error @+1 {{result components must be narrower than base components}}
  %0 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f32
  return %0 : f32
}

// -----

func.func @different_components(%base: vector<2xi32>, %offset: i32) -> vector<3xf4E2M1FN> {
  // expected-error @+1 {{base and result must have the same number of components}}
  %0 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<3xf4E2M1FN>
  return %0 : vector<3xf4E2M1FN>
}

// -----

// expected-error @+1 {{FP4 and FP6 types require Function or Private storage}}
func.func @invalid_fp4_storage(%arg: !spirv.ptr<f4E2M1FN, StorageBuffer>)

// -----

// expected-error @+1 {{FP4 and FP6 types require Function or Private storage}}
func.func @invalid_fp6e2m3_storage(%arg: !spirv.ptr<f6E2M3FN, Workgroup>)

// -----

// expected-error @+1 {{FP4 and FP6 types require Function or Private storage}}
func.func @invalid_nested_fp6_storage(%arg: !spirv.ptr<!spirv.struct<(!spirv.array<2xf6E3M2FN>)>, Uniform>)

// -----

// expected-error @+1 {{microscaling types are not supported in matrices}}
func.func @invalid_matrix(%arg: !spirv.matrix<2xvector<2xf4E2M1FN>>)

// -----

// expected-error @+1 {{microscaling types are not supported in cooperative matrices}}
func.func @invalid_coop_matrix(%arg: !spirv.coopmatrix<2x2xf4E2M1FN, Subgroup, MatrixA>)

// -----

// expected-error @+1 {{microscaling types are not supported in images}}
func.func @invalid_image(%arg: !spirv.image<f8E8M0FNU, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>)

// -----

func.func @invalid_arithmetic(%arg: f6E2M3FN) -> f6E2M3FN {
  // expected-error @+1 {{but got 'f6E2M3FN'}}
  %0 = spirv.FAdd %arg, %arg : f6E2M3FN
  return %0 : f6E2M3FN
}

// -----

func.func @invalid_dot(%arg: vector<4xf4E2M1FN>) -> f4E2M1FN {
  // expected-error @+1 {{but got 'vector<4xf4E2M1FN>'}}
  %0 = spirv.Dot %arg, %arg : vector<4xf4E2M1FN> -> f4E2M1FN
  return %0 : f4E2M1FN
}

// -----

func.func @invalid_nonuniform(%arg: f8E8M0FNU) -> f8E8M0FNU {
  %id = spirv.Constant 0 : i32
  // expected-error @+1 {{but got 'f8E8M0FNU'}}
  %0 = spirv.GroupNonUniformBroadcast <Subgroup> %arg, %id : f8E8M0FNU, i32
  return %0 : f8E8M0FNU
}

// -----

// expected-error @+1 {{microscaling types are not supported in matrices}}
func.func @invalid_matrix_result() -> !spirv.matrix<2xvector<2xf6E3M2FN>>

// -----

func.func @scalar_vector_mismatch(%base: i32, %offset: i32) -> vector<2xf6E2M3FN> {
  // expected-error @+1 {{base and result must have the same number of components}}
  %0 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to vector<2xf6E2M3FN>
  return %0 : vector<2xf6E2M3FN>
}

// -----

func.func @vector_scalar_mismatch(%base: vector<2xi32>, %offset: i32) -> f4E2M1FN {
  // expected-error @+1 {{base and result must have the same number of components}}
  %0 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to f4E2M1FN
  return %0 : f4E2M1FN
}

// -----

func.func @vector_same_width(%base: vector<2xi32>, %offset: i32) -> vector<2xf32> {
  // expected-error @+1 {{result components must be narrower than base components}}
  %0 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<2xf32>
  return %0 : vector<2xf32>
}
