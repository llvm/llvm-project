// Check for 2 thigs:
// - After round trip translation function definition has !dbg metadata attached
//   specifically if -gline-tables-only was used for Clang
// - Parent operand of DebugFunction is DebugCompileUnit, not an OpString, even
//   if in LLVM IR it points to a DIFile instead of DICompileUnit.

// RUN: %clang_cc1 %s -cl-std=c++ -emit-llvm-bc -triple spir -debug-info-kind=line-tables-only -O0 -o - | llvm-spirv -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

float foo(int i) {
    return i * 3.14;
}
void kernel k() {
    float a = foo(2);
}

// CHECK-SPIRV: String [[foo:[0-9]+]] "foo"
// CHECK-SPIRV: String [[k:[0-9]+]] "k"
// CHECK-SPIRV: [[CU:[0-9]+]] {{[0-9]+}} DebugCompileUnit
// CHECK-SPIRV: DebugFunction [[foo]] {{.*}} [[CU]] {{.*}} [[foo_id:[0-9]+]] {{[0-9]+}} {{$}}
// CHECK-SPIRV: DebugFunction [[k]] {{.*}} [[CU]] {{.*}} [[k_id:[0-9]+]] {{[0-9]+}} {{$}}

// CHECK-SPIRV: Function {{[0-9]+}} [[foo_id]]
// CHECK-LLVM: define spir_func float @_Z3fooi(i32 %i) #{{[0-9]+}} !dbg !{{[0-9]+}} {

// CHECK-SPIRV: Function {{[0-9]+}} [[k_id]]
// CHECK-LLVM: define spir_kernel void @_Z1kv() #{{[0-9]+}} !dbg !{{[0-9]+}}
