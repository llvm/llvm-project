// RUN: mlir-opt %s -test-func-erase-result -split-input-file | FileCheck %s

// CHECK: func private @f(){{$}}
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (f32 {test.erase_this_result})

// -----

// CHECK: func private @f() -> (f32 {test.A})
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  f32 {test.erase_this_result},
  f32 {test.A}
)

// -----

// CHECK: func private @f() -> (f32 {test.A})
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  f32 {test.A},
  f32 {test.erase_this_result}
)

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B})
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  f32 {test.A},
  f32 {test.erase_this_result},
  f32 {test.B}
)

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B})
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  f32 {test.A},
  f32 {test.erase_this_result},
  f32 {test.erase_this_result},
  f32 {test.B}
)

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B}, f32 {test.C})
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  f32 {test.A},
  f32 {test.erase_this_result},
  f32 {test.B},
  f32 {test.erase_this_result},
  f32 {test.C}
)

// -----

// CHECK: func private @f() -> (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>)
// CHECK-NOT: attributes{{.*}}result
func private @f() -> (
  tensor<1xf32>,
  f32 {test.erase_this_result},
  tensor<2xf32>,
  f32 {test.erase_this_result},
  tensor<3xf32>
)
