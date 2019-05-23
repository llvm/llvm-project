// Check that we can translate llvm.dbg.declare for a local variable which was
// deleted by mem2reg pass(enabled by default in llvm-spirv)

// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM


void foo() {
  int a;
}

// CHECK-SPIRV: ExtInst {{[0-9]+}} [[None:[0-9]+]] {{[0-9]+}} DebugInfoNone
// CHECK-SPIRV: ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugDeclare {{[0-9]+}} [[None]] {{[0-9]+}}
// CHECK-LLVM: call void @llvm.dbg.declare(metadata ![[VarAddr:[0-9]+]], metadata !{{[0-9]+}}, metadata !DIExpression())
// CHECK-LLVM: ![[VarAddr]] = !{}
