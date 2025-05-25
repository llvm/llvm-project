; RUN: llc < %s -mtriple=nvptx -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}

; CHECK-LABEL: .func test(
define void @test(ptr %dp0, ptr addrspace(1) %dp1, ptr addrspace(3) %dp3, double %d) {
; CHECK: atom.add.f64
  %r1 = call double @llvm.nvvm.atomic.load.add.f64.p0(ptr %dp0, double %d)
; CHECK: atom.global.add.f64
  %r2 = call double @llvm.nvvm.atomic.load.add.f64.p1(ptr addrspace(1) %dp1, double %d)
; CHECK: atom.shared.add.f64
  %ret = call double @llvm.nvvm.atomic.load.add.f64.p3(ptr addrspace(3) %dp3, double %d)
  ret void
}

; CHECK-LABEL: .func test2(
define void @test2(ptr %dp0, ptr addrspace(1) %dp1, ptr addrspace(3) %dp3, double %d) {
; CHECK: atom.add.f64
  %r1 = atomicrmw fadd ptr %dp0, double %d seq_cst
; CHECK: atom.global.add.f64
  %r2 = atomicrmw fadd ptr addrspace(1) %dp1, double %d seq_cst
; CHECK: atom.shared.add.f64
  %ret = atomicrmw fadd ptr addrspace(3) %dp3, double %d seq_cst
  ret void
}

declare double @llvm.nvvm.atomic.load.add.f64.p0(ptr nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p1(ptr addrspace(1) nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p3(ptr addrspace(3) nocapture, double) #1

attributes #1 = { argmemonly nounwind }
