; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @llvm.amdgcn.buffer.inv(i32)

define void @nonconstant(i32 %cpol) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %cpol
  ; CHECK-NEXT: call void @llvm.amdgcn.buffer.inv(i32 %cpol)
  call void @llvm.amdgcn.buffer.inv(i32 %cpol)
  ret void
}

define void @invalid_cache_policy() {
  ; CHECK: immarg value 2 for arg 0 out of range set
  ; CHECK-NEXT: call void @llvm.amdgcn.buffer.inv(i32 2)
  call void @llvm.amdgcn.buffer.inv(i32 2)
  ret void
}
