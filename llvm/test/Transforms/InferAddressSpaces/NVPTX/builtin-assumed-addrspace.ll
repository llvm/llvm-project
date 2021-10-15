; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -infer-address-spaces -o - %s | FileCheck %s

; CHECK-LABEL: @f0
; CHECK: addrspacecast float* {{%.*}} to float addrspace(4)*
; CHECK: getelementptr inbounds float, float addrspace(4)*
; CHECK: load float, float addrspace(4)*
define float @f0(float* %p) {
entry:
  %0 = bitcast float* %p to i8*
  %1 = call i1 @llvm.nvvm.isspacep.const(i8* %0)
  tail call void @llvm.assume(i1 %1)
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  ret float %3
}

; CHECK-LABEL: @f1
; CHECK: addrspacecast float* {{%.*}} to float addrspace(1)*
; CHECK: getelementptr inbounds float, float addrspace(1)*
; CHECK: load float, float addrspace(1)*
define float @f1(float* %p) {
entry:
  %0 = bitcast float* %p to i8*
  %1 = call i1 @llvm.nvvm.isspacep.global(i8* %0)
  tail call void @llvm.assume(i1 %1)
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  ret float %3
}

; CHECK-LABEL: @f2
; CHECK: addrspacecast float* {{%.*}} to float addrspace(5)*
; CHECK: getelementptr inbounds float, float addrspace(5)*
; CHECK: load float, float addrspace(5)*
define float @f2(float* %p) {
entry:
  %0 = bitcast float* %p to i8*
  %1 = call i1 @llvm.nvvm.isspacep.local(i8* %0)
  tail call void @llvm.assume(i1 %1)
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  ret float %3
}

; CHECK-LABEL: @f3
; CHECK: addrspacecast float* {{%.*}} to float addrspace(3)*
; CHECK: getelementptr inbounds float, float addrspace(3)*
; CHECK: load float, float addrspace(3)*
define float @f3(float* %p) {
entry:
  %0 = bitcast float* %p to i8*
  %1 = call i1 @llvm.nvvm.isspacep.shared(i8* %0)
  tail call void @llvm.assume(i1 %1)
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  ret float %3
}

; CHECK-LABEL: @g0
; CHECK: if.then:
; CHECK: addrspacecast float* {{%.*}} to float addrspace(3)*
; CHECK: getelementptr inbounds float, float addrspace(3)*
; CHECK: load float, float addrspace(3)*
; CHECK: if.end:
; CHECK: getelementptr inbounds float, float*
; CHECK: load float, float*
define float @g0(i32 %c, float* %p) {
entry:
  %tobool.not = icmp eq i32 %c, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %0 = bitcast float* %p to i8*
  %1 = call i1 @llvm.nvvm.isspacep.shared(i8* %0)
  tail call void @llvm.assume(i1 %1)
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idxprom = zext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  %add = fadd float %3, 0.
  br label %if.end

if.end:
  %s = phi float [ %add, %if.then ], [ 0., %entry ]
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %idxprom2 = zext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds float, float* %p, i64 %idxprom2
  %5 = load float, float* %arrayidx2, align 4
  %add2 = fadd float %s, %5
  ret float %add2
}

declare void @llvm.assume(i1)
declare i1 @llvm.nvvm.isspacep.const(i8*)
declare i1 @llvm.nvvm.isspacep.global(i8*)
declare i1 @llvm.nvvm.isspacep.local(i8*)
declare i1 @llvm.nvvm.isspacep.shared(i8*)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
