; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32"

; CHECK-LABEL: @test
; CHECK: store i32* undef, i32** %tmp9, align 8
; CHECK: store i32* undef, i32** %tmp7, align 8
define amdgpu_kernel void @test() {
entry:
  %a10.ascast.i = addrspacecast i32* addrspace(5)* null to i32**
  %tmp4 = icmp eq i32 undef, 0
  %tmp6 = select i1 false, i32** undef, i32** undef
  %tmp7 = select i1 %tmp4, i32** null, i32** %tmp6
  %tmp9 = select i1 %tmp4, i32** %a10.ascast.i, i32** null
  store i32* undef, i32** %tmp9, align 8
  store i32* undef, i32** %tmp7, align 8
  unreachable
}
