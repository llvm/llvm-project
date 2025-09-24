; RUN: opt -S -mtriple amdgcn-unknown-amdhsa -passes=instcombine < %s | FileCheck %s
; CHECK-NOT: amdgcn.mbcnt_lo
; CHECK-NOT: amdgcn.mbcnt_hi
; CHECK: @llvm.amdgcn.workitem.id.x()

define i32 @kernel() !reqd_work_group_size !0 {
entry:
  %a = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %b = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %a)
  ret i32 %b
}

!0 = !{i32 64, i32 1, i32 1}

; Declarations
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()
