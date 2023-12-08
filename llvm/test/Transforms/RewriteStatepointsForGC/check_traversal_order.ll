; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

declare void @f()
declare void @g(ptr addrspace(1), ptr addrspace(1))
declare i32 @personality_function()

; Make sure that we do not fail assertion because we process call of @g before
; we process the call of @f.

define void @test_01(ptr addrspace(1) %p, i1 %cond) gc "statepoint-example" personality ptr @personality_function {

; CHECK-LABEL: @test_01(

entry:
  %tmp0 = insertelement <2 x ptr addrspace(1)> undef, ptr addrspace(1) %p, i32 0
  %tmp1 = insertelement <2 x ptr addrspace(1)> %tmp0, ptr addrspace(1) %p, i32 1
  %tmp2 = extractelement <2 x ptr addrspace(1)> %tmp1, i32 1
  %tmp3 = extractelement <2 x ptr addrspace(1)> %tmp1, i32 0
  br label %loop

loop:
  br i1 %cond, label %cond_block, label %exit

cond_block:
  br i1 %cond, label %backedge, label %exit

exit:
  %tmp4 = phi ptr addrspace(1) [ %tmp2, %loop ], [ %tmp2, %cond_block ]
  call void @g(ptr addrspace(1) %tmp3, ptr addrspace(1) %tmp4)
  ret void

backedge:
  call void @f()
  br label %loop
}
