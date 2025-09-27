; RUN: opt -S -mtriple amdgcn-unknown-amdhsa -passes=instcombine < %s | FileCheck %s
; CHECK: llvm.amdgcn.mbcnt.lo
; CHECK: llvm.amdgcn.mbcnt.hi
; CHECK-NOT: call i32 @llvm.amdgcn.workitem.id.x()

define i32 @kernel() {
entry:
  %a = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %b = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %a)
  ret i32 %b
}

; Declarations
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
