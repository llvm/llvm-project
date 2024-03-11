; RUN: llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx63 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | FileCheck %s
; RUN: llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx62 | FileCheck %s --check-prefixes=CHECKPTX62
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx63 | %ptxas-verify -arch=sm_70 %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx63 | %ptxas-verify -arch=sm_70 %}
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_70 -mattr=+ptx62 | %ptxas-verify -arch=sm_70 %}

; CHECK-LABEL: .func test(
define void @test(ptr %dp0, ptr addrspace(1) %dp1, ptr addrspace(3) %dp3, half %val) {
; CHECK: atom.add.noftz.f16
; CHECK-NOT: atom.cas
; CHECKPTX62: atom.cas
; CHECKPTX62-NOT: atom.add.noftz.f16
  %r1 = atomicrmw fadd ptr %dp0, half %val seq_cst
; CHECK: atom.global.add.noftz.f16
; CHECK-NOT: atom.global.cas
; CHECKPTX62: atom.global.cas
; CHECKPTX62-NOT: atom.global.add.noftz.f16
  %r2 = atomicrmw fadd ptr addrspace(1) %dp1, half %val seq_cst
; CHECK: atom.shared.add.noftz.f16
; CHECK-NOT: atom.shared.cas
; CHECKPTX62: atom.shared.cas
; CHECKPTX62-NOT: atom.shared.add.noftz.f16
  %ret = atomicrmw fadd ptr addrspace(3) %dp3, half %val seq_cst
  ret void
}

attributes #1 = { argmemonly nounwind }
