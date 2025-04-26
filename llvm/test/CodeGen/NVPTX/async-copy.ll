; RUN: llc < %s -mtriple=nvptx -mcpu=sm_80 -mattr=+ptx70 | FileCheck -check-prefixes=CHECK,CHECK_PTX32 %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck -check-prefixes=CHECK,CHECK_PTX64 %s
; RUN: %if ptxas-11.0 && ! ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-11.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}

declare void @llvm.nvvm.cp.async.wait.group(i32)

; CHECK-LABEL: asyncwaitgroup
define void @asyncwaitgroup() {
  ; CHECK: cp.async.wait_group 8;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 8)
  ; CHECK: cp.async.wait_group 0;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 0)
  ; CHECK: cp.async.wait_group 16;
  tail call void @llvm.nvvm.cp.async.wait.group(i32 16)
  ret void
}

declare void @llvm.nvvm.cp.async.wait.all()

; CHECK-LABEL: asyncwaitall
define void @asyncwaitall() {
; CHECK: cp.async.wait_all
  tail call void @llvm.nvvm.cp.async.wait.all()
  ret void
}

declare void @llvm.nvvm.cp.async.commit.group()

; CHECK-LABEL: asynccommitgroup
define void @asynccommitgroup() {
; CHECK: cp.async.commit_group
  tail call void @llvm.nvvm.cp.async.commit.group()
  ret void
}

declare void @llvm.nvvm.cp.async.mbarrier.arrive(ptr %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3) %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr %a)
declare void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3) %a)

; CHECK-LABEL: asyncmbarrier
define void @asyncmbarrier(ptr %a) {
; The distinction between PTX32/PTX64 here is only to capture pointer register type
; in R to be used in subsequent tests.
; CHECK_PTX32: cp.async.mbarrier.arrive.b64 [%[[R:r]]{{[0-9]+}}];
; CHECK_PTX64: cp.async.mbarrier.arrive.b64 [%[[R:rd]]{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive(ptr %a)
  ret void
}

; CHECK-LABEL: asyncmbarriershared
define void @asyncmbarriershared(ptr addrspace(3) %a) {
; CHECK: cp.async.mbarrier.arrive.shared.b64 [%[[R]]{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3) %a)
  ret void
}

; CHECK-LABEL: asyncmbarriernoinc
define void @asyncmbarriernoinc(ptr %a) {
; CHECK_PTX64: cp.async.mbarrier.arrive.noinc.b64 [%[[R]]{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr %a)
  ret void
}

; CHECK-LABEL: asyncmbarriernoincshared
define void @asyncmbarriernoincshared(ptr addrspace(3) %a) {
; CHECK: cp.async.mbarrier.arrive.noinc.shared.b64 [%[[R]]{{[0-9]+}}];
  tail call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3) %a)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) %a, ptr addrspace(1) %b)
declare void @llvm.nvvm.cp.async.ca.shared.global.4.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)

; CHECK-LABEL: asynccasharedglobal4i8
define void @asynccasharedglobal4i8(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c) {
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 4;
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 4, %r{{[0-9]+}};
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 4, 1;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) %a, ptr addrspace(1) %b)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.4.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.4.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 1)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %a, ptr addrspace(1) %b)
declare void @llvm.nvvm.cp.async.ca.shared.global.8.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)

; CHECK-LABEL: asynccasharedglobal8i8
define void @asynccasharedglobal8i8(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c) {
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 8;
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 8, %r{{[0-9]+}};
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 8, 1;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %a, ptr addrspace(1) %b)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.8.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.8.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 1)
  ret void
}

declare void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) %a, ptr addrspace(1) %b)
declare void @llvm.nvvm.cp.async.ca.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)

; CHECK-LABEL: asynccasharedglobal16i8
define void @asynccasharedglobal16i8(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c) {
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16;
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16, %r{{[0-9]+}};
; CHECK: cp.async.ca.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16, 1;
  tail call void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) %a, ptr addrspace(1) %b)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)
  tail call void @llvm.nvvm.cp.async.ca.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 1)
  ret void
}

declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %a, ptr addrspace(1) %b)
declare void @llvm.nvvm.cp.async.cg.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)

; CHECK-LABEL: asynccgsharedglobal16i8
define void @asynccgsharedglobal16i8(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c) {
; CHECK: cp.async.cg.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16;
; CHECK: cp.async.cg.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16, %r{{[0-9]+}};
; CHECK: cp.async.cg.shared.global [%[[R]]{{[0-9]+}}], [%[[R]]{{[0-9]+}}], 16, 1;
  tail call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %a, ptr addrspace(1) %b)
  tail call void @llvm.nvvm.cp.async.cg.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 %c)
  tail call void @llvm.nvvm.cp.async.cg.shared.global.16.s(ptr addrspace(3) %a, ptr addrspace(1) %b, i32 1)
  ret void
}
