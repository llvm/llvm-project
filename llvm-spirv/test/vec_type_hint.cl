// RUN: %clang_cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-LLVM-LABEL: define spir_kernel void @test_float()
// CHECK-LLVM-SAME: !vec_type_hint [[VFLOAT:![0-9]+]]
kernel
__attribute__((vec_type_hint(float4)))
void test_float() {}

// CHECK-LLVM-LABEL: define spir_kernel void @test_double()
// CHECK-LLVM-SAME: !vec_type_hint [[VDOUBLE:![0-9]+]]
kernel
__attribute__((vec_type_hint(double)))
void test_double() {}

// CHECK-LLVM-LABEL: define spir_kernel void @test_uint()
// CHECK-LLVM-SAME: !vec_type_hint [[VUINT:![0-9]+]]
kernel
__attribute__((vec_type_hint(uint4)))
void test_uint() {}

// CHECK-LLVM-LABEL: define spir_kernel void @test_int()
// CHECK-LLVM-SAME: !vec_type_hint [[VINT:![0-9]+]]
kernel
__attribute__((vec_type_hint(int8)))
void test_int() {}

// CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_float"
// CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_double"
// CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_uint"
// CHECK-SPIRV: {{[0-9]+}} EntryPoint {{[0-9]+}} {{[0-9]+}} "test_int"
// CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
// CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
// CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}
// CHECK-SPIRV: {{[0-9]+}} ExecutionMode {{[0-9]+}} 30 {{[0-9]+}}

// CHECK-LLVM: [[VFLOAT]] = !{<4 x float> undef, i32 1}
// CHECK-LLVM: [[VDOUBLE]] = !{double undef, i32 1}
// CHECK-LLVM: [[VUINT]] = !{<4 x i32> undef, i32 1}
// CHECK-LLVM: [[VINT]] = !{<8 x i32> undef, i32 1}
