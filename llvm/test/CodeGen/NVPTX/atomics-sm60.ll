; RUN: llc < %s -march=nvptx -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_60 | FileCheck %s

; CHECK-LABEL .func test(
define void @test(double* %dp0, double addrspace(1)* %dp1, double addrspace(3)* %dp3, double %d) {
; CHECK: atom.add.f64
  %r1 = call double @llvm.nvvm.atomic.load.add.f64.p0f64(double* %dp0, double %d)
; CHECK: atom.global.add.f64
  %r2 = call double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* %dp1, double %d)
; CHECK: atom.shared.add.f64
  %ret = call double @llvm.nvvm.atomic.load.add.f64.p3f64(double addrspace(3)* %dp3, double %d)
  ret void
}

declare double @llvm.nvvm.atomic.load.add.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.load.add.f64.p3f64(double addrspace(3)* nocapture, double) #1

attributes #1 = { argmemonly nounwind }
