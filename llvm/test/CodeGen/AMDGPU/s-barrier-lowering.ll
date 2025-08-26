; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -verify-machineinstrs -o - %s | FileCheck -check-prefixes=SOUT %s

%class.ExpAmdWorkgroupWaveBarrier = type { target("amdgcn.named.barrier", 0) }

@bar2 = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison
@bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@bar1 = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison

; CHECK: @bar2 = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison, !absolute_symbol !0
; CHECK-NEXT: @bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !1
; CHECK-NEXT: @bar1 = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison, !absolute_symbol !2
; CHECK-NEXT: @bar1.kernel1 = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison, !absolute_symbol !2

; SOUT:        .set func1.num_named_barrier, 7
define void @func1() {
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar3, i32 7)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar3)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

; SOUT:        .set func2.num_named_barrier, 2
define void @func2() {
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar2, i32 7)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar2)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

; SOUT:                .amdhsa_named_barrier_count 2
; SOUT:        .set kernel1.num_named_barrier, max(6, func1.num_named_barrier, func2.num_named_barrier)
define amdgpu_kernel void @kernel1() #0 {
; CHECK-DAG: call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1.kernel1, i32 11)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 11)
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) @bar1)
    call void @llvm.amdgcn.s.barrier()
    call void @func1()
    call void @func2()
    ret void
}

; SOUT:                .amdhsa_named_barrier_count 2
; SOUT:        .set kernel2.num_named_barrier, max(6, func2.num_named_barrier)
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
; CHECK-NEXT: !1 = !{i32 8396912, i32 8396913}
; CHECK-NEXT: !2 = !{i32 8396848, i32 8396849}
