; RUN: llc < %s -march=nvptx -mcpu=sm_80 -mattr=+ptx70 | FileCheck -check-prefixes=ALL,CHECK_PTX32 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck -check-prefixes=ALL,CHECK_PTX64 %s
; RUN: %if ptxas-11.0 %{ llc < %s -march=nvptx -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-11.0 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}

declare void @llvm.nvvm.cp.async.wait.group(i32)

; ALL-LABEL: asyncwaitgroup
define void @asyncwaitgroup() {
  ; ALL: cp.async.wait_group 8;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 8)
  ; ALL: cp.async.wait_group 0;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 0)
  ; ALL: cp.async.wait_group 16;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 16)
  ret void
}

declare void @llvm.nvvm.cp.async.wait.all()

; ALL-LABEL: asyncwaitall
define void @asyncwaitall() {
; ALL: cp.async.wait_all
  tail call void @llvm.nvvm.cp.async.wait.all()
  ret void
}

declare void @llvm.nvvm.cp.async.commit.group()

; ALL-LABEL: asynccommitgroup
define void @asynccommitgroup() {
; ALL: cp.async.commit_group
  tail call void @llvm.nvvm.cp.async.commit.group()
  ret void
}

declare void @llvm.nvvm.cp.async.mbarrier.arrive(i64* %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.shared(i64 addrspace(3)* %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(i64* %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(i64 addrspace(3)* %a)

; CHECK-LABEL: asyncmbarrier
define void @asyncmbarrier(i64* %a) {
; CHECK_PTX32: cp.async.mbarrier.arrive.b64 [%r{{[0-9]+}}];
; CHECK_PTX64: cp.async.mbarrier.arrive.b64 [%rd{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive(i64* %a)
  ret void
}

; CHECK-LABEL: asyncmbarriershared
define void @asyncmbarriershared(i64 addrspace(3)* %a) {
; CHECK_PTX32: cp.async.mbarrier.arrive.shared.b64 [%r{{[0-9]+}}];
; CHECK_PTX64: cp.async.mbarrier.arrive.shared.b64 [%rd{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.shared(i64 addrspace(3)* %a)
  ret void
}

; CHECK-LABEL: asyncmbarriernoinc
define void @asyncmbarriernoinc(i64* %a) {
; CHECK_PTX32: cp.async.mbarrier.arrive.noinc.b64 [%r{{[0-9]+}}];
; CHECK_PTX64: cp.async.mbarrier.arrive.noinc.b64 [%rd{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(i64* %a)
  ret void
}

; CHECK-LABEL: asyncmbarriernoincshared
define void @asyncmbarriernoincshared(i64 addrspace(3)* %a) {
; CHECK_PTX32: cp.async.mbarrier.arrive.noinc.shared.b64 [%r{{[0-9]+}}];
; CHECK_PTX64: cp.async.mbarrier.arrive.noinc.shared.b64 [%rd{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(i64 addrspace(3)* %a)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.4(i8 addrspace(3)* %a, i8 addrspace(1)* %b)

; CHECK-LABEL: asynccasharedglobal4i8
define void @asynccasharedglobal4i8(i8 addrspace(3)* %a, i8 addrspace(1)* %b) {
; CHECK_PTX32: cp.async.ca.shared.global [%r{{[0-9]+}}], [%r{{[0-9]+}}], 4;
; CHECK_PTX64: cp.async.ca.shared.global [%rd{{[0-9]+}}], [%rd{{[0-9]+}}], 4;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.4(i8 addrspace(3)* %a, i8 addrspace(1)* %b)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.8(i8 addrspace(3)* %a, i8 addrspace(1)* %b)

; CHECK-LABEL: asynccasharedglobal8i8
define void @asynccasharedglobal8i8(i8 addrspace(3)* %a, i8 addrspace(1)* %b) {
; CHECK_PTX32: cp.async.ca.shared.global [%r{{[0-9]+}}], [%r{{[0-9]+}}], 8;
; CHECK_PTX64: cp.async.ca.shared.global [%rd{{[0-9]+}}], [%rd{{[0-9]+}}], 8;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.8(i8 addrspace(3)* %a, i8 addrspace(1)* %b)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.16(i8 addrspace(3)* %a, i8 addrspace(1)* %b)

; CHECK-LABEL: asynccasharedglobal16i8
define void @asynccasharedglobal16i8(i8 addrspace(3)* %a, i8 addrspace(1)* %b) {
; CHECK_PTX32: cp.async.ca.shared.global [%r{{[0-9]+}}], [%r{{[0-9]+}}], 16;
; CHECK_PTX64: cp.async.ca.shared.global [%rd{{[0-9]+}}], [%rd{{[0-9]+}}], 16;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.16(i8 addrspace(3)* %a, i8 addrspace(1)* %b)
  ret void
}

declare void @llvm.nvvm.cp.async.cg.shared.global.16(i8 addrspace(3)* %a, i8 addrspace(1)* %b)

; CHECK-LABEL: asynccgsharedglobal16i8
define void @asynccgsharedglobal16i8(i8 addrspace(3)* %a, i8 addrspace(1)* %b) {
; CHECK_PTX32: cp.async.cg.shared.global [%r{{[0-9]+}}], [%r{{[0-9]+}}], 16;
; CHECK_PTX64: cp.async.cg.shared.global [%rd{{[0-9]+}}], [%rd{{[0-9]+}}], 16;
  tail call void @llvm.nvvm.cp.async.cg.shared.global.16(i8 addrspace(3)* %a, i8 addrspace(1)* %b)
  ret void
}
