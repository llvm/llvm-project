; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define i32 @user(i32 %a, i32 %b, i32 %c) {
  ; CHECK: %call = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  ; CHECK-NOT: amdgcn.alignbit
  %call = call i32 @llvm.amdgcn.alignbit(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}

declare i32 @llvm.amdgcn.alignbit(i32, i32, i32)
; CHECK: declare i32 @llvm.fshr.i32(i32, i32, i32) #0
; CHECK-NOT: amdgcn.alignbit
