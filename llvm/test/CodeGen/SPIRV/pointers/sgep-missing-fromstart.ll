; RUN: not llc -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: llvm.structured.gep requires fromstart on every index for SPIR-V lowering

define ptr @missing_fromstart(ptr %p, i32 %idx) {
entry:
  %q = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([10 x i32]) %p, <1 x i32> <i32 1>, i32 %idx)
  ret ptr %q
}

declare ptr @llvm.structured.gep.p0.v1i32(ptr, <1 x i32>, ...) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
