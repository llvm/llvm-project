; RUN: opt -aa-pipeline=globals-aa -passes='require<globals-aa>,gvn' -S < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa,globals-aa -passes='require<globals-aa>,gvn' -S < %s | FileCheck %s
;
; Functions w/o `nosync` attribute may communicate via memory and must be
; treated conservatively. Taken from https://reviews.llvm.org/D115302.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@s = internal local_unnamed_addr addrspace(3) global i32 undef, align 4

; CHECK-LABEL: @bar_sync
; CHECK: store
; CHECK: tail call void @llvm.nvvm.bar.sync(i32 0)
; CHECK: load
define dso_local i32 @bar_sync(i32 %0) local_unnamed_addr {
  store i32 %0, i32* addrspacecast (i32 addrspace(3)* @s to i32*), align 4
  tail call void @llvm.nvvm.bar.sync(i32 0)
  %2 = load i32, i32* addrspacecast (i32 addrspace(3)* @s to i32*), align 4
  ret i32 %2
}

declare void @llvm.nvvm.bar.sync(i32) #0

; CHECK-LABEL: @barrier0
; CHECK: store
; CHECK: tail call void @llvm.nvvm.barrier0()
; CHECK: load
define dso_local i32 @barrier0(i32 %0) local_unnamed_addr  {
  store i32 %0, i32* addrspacecast (i32 addrspace(3)* @s to i32*), align 4
  tail call void @llvm.nvvm.barrier0()
  %2 = load i32, i32* addrspacecast (i32 addrspace(3)* @s to i32*), align 4
  ret i32 %2
}

declare void @llvm.nvvm.barrier0() #0

attributes #0 = { convergent nounwind }
