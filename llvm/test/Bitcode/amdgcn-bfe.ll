; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; ubfe/sbfe are no longer overloaded; check that the legacy mangled .i32 names
; are auto-upgraded to the unmangled form.

define i32 @ubfe(i32 %src, i32 %offset, i32 %width) {
  ; CHECK: %call = call i32 @llvm.amdgcn.ubfe(i32 %src, i32 %offset, i32 %width)
  ; CHECK-NOT: amdgcn.ubfe.i32
  %call = call i32 @llvm.amdgcn.ubfe.i32(i32 %src, i32 %offset, i32 %width)
  ret i32 %call
}

define i32 @sbfe(i32 %src, i32 %offset, i32 %width) {
  ; CHECK: %call = call i32 @llvm.amdgcn.sbfe(i32 %src, i32 %offset, i32 %width)
  ; CHECK-NOT: amdgcn.sbfe.i32
  %call = call i32 @llvm.amdgcn.sbfe.i32(i32 %src, i32 %offset, i32 %width)
  ret i32 %call
}

declare i32 @llvm.amdgcn.ubfe.i32(i32, i32, i32)
declare i32 @llvm.amdgcn.sbfe.i32(i32, i32, i32)
; CHECK: declare i32 @llvm.amdgcn.ubfe(i32, i32, i32)
; CHECK: declare i32 @llvm.amdgcn.sbfe(i32, i32, i32)
; CHECK-NOT: amdgcn.ubfe.i32
; CHECK-NOT: amdgcn.sbfe.i32
