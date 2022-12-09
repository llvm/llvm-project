; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces -o - %s | FileCheck %s

; CHECK-LABEL: @f0
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(3)
; CHECK: getelementptr inbounds float, ptr addrspace(3)
; CHECK: load float, ptr addrspace(3)
define float @f0(ptr %p) {
entry:
  %0 = call i1 @llvm.amdgcn.is.shared(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @f1
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(5)
; CHECK: getelementptr inbounds float, ptr addrspace(5)
; CHECK: load float, ptr addrspace(5)
define float @f1(ptr %p) {
entry:
  %0 = call i1 @llvm.amdgcn.is.private(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  ret float %2
}

; CHECK-LABEL: @f2
; CHECK: addrspacecast ptr {{%.*}} to ptr addrspace(1)
; CHECK: getelementptr inbounds float, ptr addrspace(1)
; CHECK: load float, ptr addrspace(1)
define float @f2(ptr %p) {
entry:
  %0 = call i1 @llvm.amdgcn.is.private(ptr %p)
  %1 = xor i1 %0, -1
  %2 = call i1 @llvm.amdgcn.is.shared(ptr %p)
  %3 = xor i1 %2, -1
  %4 = and i1 %1, %3
  tail call void @llvm.assume(i1 %4)
  %5 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %5 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %6 = load float, ptr %arrayidx, align 4
  ret float %6
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
  %0 = call i1 @llvm.amdgcn.is.shared(ptr %p)
  tail call void @llvm.assume(i1 %0)
  %1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %idxprom = zext i32 %1 to i64
  %arrayidx = getelementptr inbounds float, ptr %p, i64 %idxprom
  %2 = load float, ptr %arrayidx, align 4
  %add = fadd float %2, 0.
  br label %if.end

if.end:
  %s = phi float [ %add, %if.then ], [ 0., %entry ]
  %3 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %idxprom2 = zext i32 %3 to i64
  %arrayidx2 = getelementptr inbounds float, ptr %p, i64 %idxprom2
  %4 = load float, ptr %arrayidx2, align 4
  %add2 = fadd float %s, %4
  ret float %add2
}

declare void @llvm.assume(i1)
declare i1 @llvm.amdgcn.is.shared(ptr nocapture)
declare i1 @llvm.amdgcn.is.private(ptr nocapture)
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
