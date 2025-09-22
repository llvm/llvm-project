; RUN: llc < %s -mtriple=nvptx -mcpu=sm_80 | FileCheck %s -check-prefix=CHECK_PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 | FileCheck %s -check-prefix=CHECK_PTX64
; RUN: %if ptxas-sm_80 && ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-sm_80 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}

declare void @llvm.nvvm.mbarrier.init(ptr %a, i32 %b)
declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) %a, i32 %b)

; CHECK-LABEL: barrierinit
define void @barrierinit(ptr %a, i32 %b) {
; CHECK_PTX32: mbarrier.init.b64 [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.init.b64 [%rd{{[0-9]+}}], %r{{[0-9]+}};
  tail call void @llvm.nvvm.mbarrier.init(ptr %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierinitshared
define void @barrierinitshared(ptr addrspace(3) %a, i32 %b) {
; CHECK_PTX32: mbarrier.init.shared.b64 [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.init.shared.b64 [%rd{{[0-9]+}}], %r{{[0-9]+}};
  tail call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) %a, i32 %b)
  ret void
}

declare void @llvm.nvvm.mbarrier.inval(ptr %a)
declare void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) %a)

; CHECK-LABEL: barrierinval
define void @barrierinval(ptr %a) {
; CHECK_PTX32: mbarrier.inval.b64 [%r{{[0-1]+}}];
; CHECK_PTX64: mbarrier.inval.b64 [%rd{{[0-1]+}}];
  tail call void @llvm.nvvm.mbarrier.inval(ptr %a)
  ret void
}

; CHECK-LABEL: barrierinvalshared
define void @barrierinvalshared(ptr addrspace(3) %a) {
; CHECK_PTX32: mbarrier.inval.shared.b64 [%r{{[0-1]+}}];
; CHECK_PTX64: mbarrier.inval.shared.b64 [%rd{{[0-1]+}}];
  tail call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive(ptr %a)
declare i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %a)

; CHECK-LABEL: barrierarrive
define void @barrierarrive(ptr %a) {
; CHECK_PTX32: mbarrier.arrive.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive(ptr %a)
  ret void
}

; CHECK-LABEL: barrierarriveshared
define void @barrierarriveshared(ptr addrspace(3) %a) {
; CHECK_PTX32: mbarrier.arrive.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.noComplete(ptr %a, i32 %b)
declare i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(ptr addrspace(3) %a, i32 %b)

; CHECK-LABEL: barrierarrivenoComplete
define void @barrierarrivenoComplete(ptr %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive.noComplete.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive.noComplete.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.noComplete(ptr %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierarrivenoCompleteshared
define void @barrierarrivenoCompleteshared(ptr addrspace(3) %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive.noComplete.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive.noComplete.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(ptr addrspace(3) %a, i32 %b)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.drop(ptr %a)
declare i64 @llvm.nvvm.mbarrier.arrive.drop.shared(ptr addrspace(3) %a)

; CHECK-LABEL: barrierarrivedrop
define void @barrierarrivedrop(ptr %a) {
; CHECK_PTX32: mbarrier.arrive_drop.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive_drop.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop(ptr %a)
  ret void
}

; CHECK-LABEL: barrierarrivedropshared
define void @barrierarrivedropshared(ptr addrspace(3) %a) {
; CHECK_PTX32: mbarrier.arrive_drop.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive_drop.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.shared(ptr addrspace(3) %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete(ptr %a, i32 %b)
declare i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared(ptr addrspace(3) %a, i32 %b)

; CHECK-LABEL: barrierarrivedropnoComplete
define void @barrierarrivedropnoComplete(ptr %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive_drop.noComplete.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive_drop.noComplete.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete(ptr %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierarrivedropnoCompleteshared
define void @barrierarrivedropnoCompleteshared(ptr addrspace(3) %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive_drop.noComplete.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive_drop.noComplete.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared(ptr addrspace(3) %a, i32 %b)
  ret void
}

declare i1 @llvm.nvvm.mbarrier.test.wait(ptr %a, i64 %b)
declare i1 @llvm.nvvm.mbarrier.test.wait.shared(ptr addrspace(3) %a, i64 %b)

; CHECK-LABEL: barriertestwait
define void @barriertestwait(ptr %a, i64 %b) {
; CHECK_PTX32: mbarrier.test_wait.b64 %p{{[0-9]+}}, [%r{{[0-9]+}}], %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.test_wait.b64 %p{{[0-9]+}}, [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  %ret = tail call i1 @llvm.nvvm.mbarrier.test.wait(ptr %a, i64 %b)
  ret void
}

; CHECK-LABEL: barriertestwaitshared
define void @barriertestwaitshared(ptr addrspace(3) %a, i64 %b) {
; CHECK_PTX32: mbarrier.test_wait.shared.b64 %p{{[0-9]+}}, [%r{{[0-9]+}}], %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.test_wait.shared.b64 %p{{[0-9]+}}, [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  %ret = tail call i1 @llvm.nvvm.mbarrier.test.wait.shared(ptr addrspace(3) %a, i64 %b)
  ret void
}

declare i32 @llvm.nvvm.mbarrier.pending.count(i64 %b)

; CHECK-LABEL: barrierpendingcount
define i32 @barrierpendingcount(ptr %a, i64 %b) {
; CHECK_PTX32: mbarrier.pending_count.b64 %r{{[0-9]+}}, %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.pending_count.b64 %r{{[0-9]+}}, %rd{{[0-9]+}};
  %ret = tail call i32 @llvm.nvvm.mbarrier.pending.count(i64 %b)
  ret i32 %ret
}
