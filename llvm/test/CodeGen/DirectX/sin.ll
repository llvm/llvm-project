// Shader Mode 6.0
// RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %S/Inputs/sin/half.ll 2>&1 | FileCheck %s -check-prefix=SM6_0_HALF
// RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %S/Inputs/sin/float.ll | FileCheck %s -check-prefix=SM6_0_FLOAT
// RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %S/inputs/sin/double.ll 2>&1 | FileCheck %s --check-prefix=SM6_0_DOUBLE

// RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %S/Inputs/sin/half.ll | FileCheck %s -check-prefix=SM6_3_HALF
// RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %S/Inputs/sin/float.ll | FileCheck %s -check-prefix=SM6_3_FLOAT
// RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %S/inputs/sin/double.ll 2>&1 | FileCheck %s --check-prefix=SM6_3_DOUBLE

// Float is valid for SM6.0
// SM6_0_FLOAT: call float @dx.op.unary.f32(i32 13, float %{{.*}})

// Half is not valid for SM6.0
// SM6_0_HALF: LLVM ERROR: Invalid Overload

// Half and float are valid for SM6.2 and later
// SM6_3_HALF: call half @dx.op.unary.f16(i32 13, half %{{.*}})
// SM6_3_FLOAT: call float @dx.op.unary.f32(i32 13, float %{{.*}})

// Double is not valid in any Shader Model version
// SM6_0_DOUBLE: LLVM ERROR: Invalid Overload
// SM6_3_DOUBLE: LLVM ERROR: Invalid Overload

