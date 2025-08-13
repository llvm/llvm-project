; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s

@bar2 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@bar1 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison

; CHECK: @bar2 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !0
; CHECK-NEXT: @bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !1
; CHECK-NEXT: @bar1 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !2
; CHECK-NEXT: @bar1.kernel1 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !2

define void @func1() {
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar3, i32 7)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar3)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

define void @func2() {
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar2, i32 7)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar2)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

define amdgpu_kernel void @kernel1() #0 {
; CHECK-DAG: call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1.kernel1, i32 11)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 11)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    call void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3) @bar1)
    %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) @bar1)
    call void @llvm.amdgcn.s.barrier()
    call void @func1()
    call void @func2()
    ret void
}

define amdgpu_kernel void @kernel2() #0 {
; CHECK-DAG: call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 9)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 9)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)

    call void @func2()
    ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare void @llvm.amdgcn.s.barrier.wait(i16) #1
declare void @llvm.amdgcn.s.barrier.signal(i32) #1
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32) #1
declare i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32) #1
declare void @llvm.amdgcn.s.barrier.init(ptr addrspace(3), i32) #1
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3)) #1
declare void @llvm.amdgcn.s.barrier.leave(i16) #1
declare void @llvm.amdgcn.s.wakeup.barrier(ptr addrspace(3)) #1
declare i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3)) #1

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }

; CHECK: !0 = !{i32 8396816, i32 8396817}
; CHECK-NEXT: !1 = !{i32 8396848, i32 8396849}
; CHECK-NEXT: !2 = !{i32 8396832, i32 8396833}
