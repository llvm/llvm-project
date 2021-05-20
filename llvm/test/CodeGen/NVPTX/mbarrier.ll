; RUN: llc < %s -march=nvptx -mcpu=sm_80 | FileCheck %s -check-prefix=CHECK_PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 | FileCheck %s -check-prefix=CHECK_PTX64

declare void @llvm.nvvm.mbarrier.init(i64* %a, i32 %b)
declare void @llvm.nvvm.mbarrier.init.shared(i64 addrspace(3)* %a, i32 %b)

; CHECK-LABEL: barrierinit
define void @barrierinit(i64* %a, i32 %b) {
; CHECK_PTX32: mbarrier.init.b64 [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.init.b64 [%rd{{[0-9]+}}], %r{{[0-9]+}};
  tail call void @llvm.nvvm.mbarrier.init(i64* %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierinitshared
define void @barrierinitshared(i64 addrspace(3)* %a, i32 %b) {
; CHECK_PTX32: mbarrier.init.shared.b64 [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.init.shared.b64 [%rd{{[0-9]+}}], %r{{[0-9]+}};
  tail call void @llvm.nvvm.mbarrier.init.shared(i64 addrspace(3)* %a, i32 %b)
  ret void
}

declare void @llvm.nvvm.mbarrier.inval(i64* %a)
declare void @llvm.nvvm.mbarrier.inval.shared(i64 addrspace(3)* %a)

; CHECK-LABEL: barrierinval
define void @barrierinval(i64* %a) {
; CHECK_PTX32: mbarrier.inval.b64 [%r{{[0-1]+}}];
; CHECK_PTX64: mbarrier.inval.b64 [%rd{{[0-1]+}}];
  tail call void @llvm.nvvm.mbarrier.inval(i64* %a)
  ret void
}

; CHECK-LABEL: barrierinvalshared
define void @barrierinvalshared(i64 addrspace(3)* %a) {
; CHECK_PTX32: mbarrier.inval.shared.b64 [%r{{[0-1]+}}];
; CHECK_PTX64: mbarrier.inval.shared.b64 [%rd{{[0-1]+}}];
  tail call void @llvm.nvvm.mbarrier.inval.shared(i64 addrspace(3)* %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive(i64* %a)
declare i64 @llvm.nvvm.mbarrier.arrive.shared(i64 addrspace(3)* %a)

; CHECK-LABEL: barrierarrive
define void @barrierarrive(i64* %a) {
; CHECK_PTX32: mbarrier.arrive.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive(i64* %a)
  ret void
}

; CHECK-LABEL: barrierarriveshared
define void @barrierarriveshared(i64 addrspace(3)* %a) {
; CHECK_PTX32: mbarrier.arrive.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.shared(i64 addrspace(3)* %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.noComplete(i64* %a, i32 %b)
declare i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(i64 addrspace(3)* %a, i32 %b)

; CHECK-LABEL: barrierarrivenoComplete
define void @barrierarrivenoComplete(i64* %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive.noComplete.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive.noComplete.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.noComplete(i64* %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierarrivenoCompleteshared
define void @barrierarrivenoCompleteshared(i64 addrspace(3)* %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive.noComplete.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive.noComplete.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(i64 addrspace(3)* %a, i32 %b)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.drop(i64* %a)
declare i64 @llvm.nvvm.mbarrier.arrive.drop.shared(i64 addrspace(3)* %a)

; CHECK-LABEL: barrierarrivedrop
define void @barrierarrivedrop(i64* %a) {
; CHECK_PTX32: mbarrier.arrive_drop.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive_drop.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop(i64* %a)
  ret void
}

; CHECK-LABEL: barrierarrivedropshared
define void @barrierarrivedropshared(i64 addrspace(3)* %a) {
; CHECK_PTX32: mbarrier.arrive_drop.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}];
; CHECK_PTX64: mbarrier.arrive_drop.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}];
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.shared(i64 addrspace(3)* %a)
  ret void
}

declare i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete(i64* %a, i32 %b)
declare i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared(i64 addrspace(3)* %a, i32 %b)

; CHECK-LABEL: barrierarrivedropnoComplete
define void @barrierarrivedropnoComplete(i64* %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive_drop.noComplete.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive_drop.noComplete.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete(i64* %a, i32 %b)
  ret void
}

; CHECK-LABEL: barrierarrivedropnoCompleteshared
define void @barrierarrivedropnoCompleteshared(i64 addrspace(3)* %a, i32 %b) {
; CHECK_PTX32: mbarrier.arrive_drop.noComplete.shared.b64 %rd{{[0-9]+}}, [%r{{[0-9]+}}], %r{{[0-9]+}};
; CHECK_PTX64: mbarrier.arrive_drop.noComplete.shared.b64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}], %r{{[0-9]+}};
  %ret = tail call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared(i64 addrspace(3)* %a, i32 %b)
  ret void
}

declare i1 @llvm.nvvm.mbarrier.test.wait(i64* %a, i64 %b)
declare i1 @llvm.nvvm.mbarrier.test.wait.shared(i64 addrspace(3)* %a, i64 %b)

; CHECK-LABEL: barriertestwait
define void @barriertestwait(i64* %a, i64 %b) {
; CHECK_PTX32: mbarrier.test_wait.b64 %p{{[0-9]+}}, [%r{{[0-9]+}}], %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.test_wait.b64 %p{{[0-9]+}}, [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  %ret = tail call i1 @llvm.nvvm.mbarrier.test.wait(i64* %a, i64 %b)
  ret void
}

; CHECK-LABEL: barriertestwaitshared
define void @barriertestwaitshared(i64 addrspace(3)* %a, i64 %b) {
; CHECK_PTX32: mbarrier.test_wait.shared.b64 %p{{[0-9]+}}, [%r{{[0-9]+}}], %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.test_wait.shared.b64 %p{{[0-9]+}}, [%rd{{[0-9]+}}], %rd{{[0-9]+}};
  %ret = tail call i1 @llvm.nvvm.mbarrier.test.wait.shared(i64 addrspace(3)* %a, i64 %b)
  ret void
}

declare i32 @llvm.nvvm.mbarrier.pending.count(i64 %b)

; CHECK-LABEL: barrierpendingcount
define i32 @barrierpendingcount(i64* %a, i64 %b) {
; CHECK_PTX32: mbarrier.pending_count.b64 %r{{[0-9]+}}, %rd{{[0-9]+}};
; CHECK_PTX64: mbarrier.pending_count.b64 %r{{[0-9]+}}, %rd{{[0-9]+}};
  %ret = tail call i32 @llvm.nvvm.mbarrier.pending.count(i64 %b)
  ret i32 %ret
}
