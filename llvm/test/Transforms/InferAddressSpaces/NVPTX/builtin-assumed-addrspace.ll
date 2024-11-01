; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes=infer-address-spaces -o - %s | FileCheck %s

; CHECK-LABEL: @f0
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(4)
; CHECK: getelementptr inbounds float, ptr addrspace(4)
; CHECK: load float, ptr addrspace(4)
define float @f0(ptr %p) {
entry:
  %0 = call i1 @llvm.nvvm.isspacep.const(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @f1
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(1)
; CHECK: getelementptr inbounds float, ptr addrspace(1)
; CHECK: load float, ptr addrspace(1)
define float @f1(ptr %p) {
entry:
  %0 = call i1 @llvm.nvvm.isspacep.global(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @f2
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(5)
; CHECK: getelementptr inbounds float, ptr addrspace(5)
; CHECK: load float, ptr addrspace(5)
define float @f2(ptr %p) {
entry:
  %0 = call i1 @llvm.nvvm.isspacep.local(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @f3
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(3)
; CHECK: getelementptr inbounds float, ptr addrspace(3)
; CHECK: load float, ptr addrspace(3)
define float @f3(ptr %p) {
entry:
  %0 = call i1 @llvm.nvvm.isspacep.shared(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @g0
; CHECK: if.then:
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(3)
; CHECK: getelementptr inbounds float, ptr addrspace(3)
; CHECK: load float, ptr addrspace(3)
; CHECK: if.end:
; CHECK: getelementptr inbounds float, ptr
; CHECK: load float, ptr
define float @g0(i32 %c, ptr %p) {
entry:
  %tobool.not = icmp eq i32 %c, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %0 = call i1 @llvm.nvvm.isspacep.shared(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  %add = fadd float %2, 0.
  br label %if.end

if.end:
  %s = phi float [ %add, %if.then ], [ 0., %entry ]
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom2 = zext i32 %3 to i64
  %arrayidx2 = getelementptr inbounds float, ptr %p, i64 %idxprom2
  %4 = load float, ptr %arrayidx2, align 4
  %add2 = fadd float %s, %4
  ret float %add2
}

declare void @llvm.assume(i1)
declare i1 @llvm.nvvm.isspacep.const(ptr)
declare i1 @llvm.nvvm.isspacep.global(ptr)
declare i1 @llvm.nvvm.isspacep.local(ptr)
declare i1 @llvm.nvvm.isspacep.shared(ptr)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
