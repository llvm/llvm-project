; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=instcombine < %s | FileCheck %s
; CHECK-NOT: and i32
; CHECK-NOT: @llvm.amdgcn.workitem.id.x()

; ModuleID = 'mbcnt_to_bitmask_neg'

define i32 @kernel() !reqd_work_group_size !1 {
entry:
  %a = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %b = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %a)
  ret i32 %b
}

!1 = !{i32 48, i32 1, i32 1}

; Declarations
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
