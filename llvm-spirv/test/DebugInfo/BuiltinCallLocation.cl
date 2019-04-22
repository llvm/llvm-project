// Check that DebugLoc attached to a builtin call is preserved after translation.

// RUN: %clang_cc1 -triple spir -finclude-default-header %s -disable-llvm-passes -emit-llvm-bc -debug-info-kind=line-tables-only -dwarf-column-info -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: Label
// CHECK-SPIRV: ExtInst {{.*}} DebugScope
// CHECK-SPIRV: ExtInst {{.*}} sin
// CHECK-LLVM: call spir_func float @_Z3sinf(float %x) {{.*}} !dbg ![[loc:[0-9]+]]
// CHECK-LLVM: ![[loc]] = !DILocation(line: 14, column: 10, scope: !{{.*}})
float f(float x) {
  return sin(x);
}
