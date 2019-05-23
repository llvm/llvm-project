// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 -O0 -debug-info-kind=standalone -emit-llvm %s -o %t.ll
// RUN: llvm-as %t.ll -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Test that no debug info instruction is inserted
// between LoopMerge and Branch/BranchConditional instructions.
// Otherwise, debug info interferes with SPIRVToLLVM translation
// of structured flow control
//
// Currently, Line DebugInfo instructions are still present
// between LoopMerge and Branch/BranchConditional instructions.
// This does not affect SPIRVToLLVM translation, however
// should be fixed separately

kernel
void sample() {
  int arr[10];
  #pragma clang loop unroll(full)
  for (int i = 0; i < 10; i++)
    arr[i] = 0;
  int j = 0;
  #pragma clang loop unroll(full)
  do {
    arr[j] = 0;
  } while (j++ < 10);
}

// CHECK-SPIRV: {{[0-9]+}} LoopMerge [[MergeBlock:[0-9]+]] [[ContinueTarget:[0-9]+]] 1
// CHECK-SPIRV-NOT: ExtInst
// CHECK-SPIRV: BranchConditional
// CHECK-SPIRV: {{[0-9]+}} LoopMerge [[MergeBlock:[0-9]+]] [[ContinueTarget:[0-9]+]] 1
// CHECK-SPIRV-NOT: ExtInst
// CHECK-SPIRV: Branch
// CHECK-LLVM: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !dbg !{{[0-9]+}}, !llvm.loop ![[MD:[0-9]+]]
// CHECK-LLVM: ![[MD]] = distinct !{![[MD]], ![[MD_unroll:[0-9]+]]}
// CHECK-LLVM: ![[MD_unroll]] = !{!"llvm.loop.unroll.enable"}
