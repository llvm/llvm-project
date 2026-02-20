; RUN: llc < %s -mtriple=nvptx -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas-sm_60 && ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}
; RUN: %if ptxas-sm_60 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}

; CHECK-LABEL: .func test(
define void @test(ptr %dp0, ptr addrspace(1) %dp1, ptr addrspace(3) %dp3, double %d) {
; CHECK: atom.sys.add.f64
  %r1 = call double @llvm.nvvm.atomic.load.add.f64.p0(ptr %dp0, double %d)
; CHECK: atom.sys.global.add.f64
  %r2 = call double @llvm.nvvm.atomic.load.add.f64.p1(ptr addrspace(1) %dp1, double %d)
; CHECK: atom.sys.shared.add.f64
  %ret = call double @llvm.nvvm.atomic.load.add.f64.p3(ptr addrspace(3) %dp3, double %d)
  ret void
}

declare double @llvm.nvvm.atomic.load.add.f64.p0(ptr nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p1(ptr addrspace(1) nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p3(ptr addrspace(3) nocapture, double) #1

attributes #1 = { argmemonly nounwind }
