; RUN: opt -S -mtriple=amdgpu-- -passes=amdgpu-lower-exec-sync,amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs -o - %s | FileCheck -check-prefixes=SOUT %s

%class.ExpAmdWorkgroupWaveBarrier = type { target("amdgcn.named.barrier", 0) }

@bar2 = internal addrspace(15) global [2 x target("amdgcn.named.barrier", 0)] poison
@bar3 = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison
@bar1 = internal addrspace(15) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison

; Test using the workgroup barrier with the GV.
@wgbarr = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol !0

; CHECK: @bar2 = internal addrspace(15) global [2 x target("amdgcn.named.barrier", 0)] poison, !absolute_symbol [[META0:![0-9]+]]
; CHECK-NEXT: @bar3 = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol [[META1:![0-9]+]]
; CHECK-NEXT: @bar1 = internal addrspace(15) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison, !absolute_symbol [[META2:![0-9]+]]
; CHECK-NEXT: @wgbarr = internal addrspace(15) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol [[META3:![0-9]+]]

; SOUT:        .set .Lfunc1.num_named_barrier, 7
define void @func1() {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar3)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar3, i32 7)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

; SOUT:        .set .Lfunc2.num_named_barrier, 6
define void @func2() {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar2)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar2, i32 7)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    ret void
}

; SOUT:                .amdhsa_named_barrier_count 2
; SOUT:        .set .Lkernel1.num_named_barrier, max(4, .Lfunc1.num_named_barrier, .Lfunc2.num_named_barrier)
define amdgpu_kernel void @kernel1() #0 {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar1)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar1, i32 11)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)
    %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(15) @bar1)
    call void @llvm.amdgcn.s.barrier()
    call void @func1()
    call void @func2()
    ret void
}

; SOUT:      .amdhsa_kernel kernel2
; SOUT:        .amdhsa_named_barrier_count 2
; SOUT:        .set .Lkernel2.num_named_barrier, max(4, .Lfunc2.num_named_barrier)
define amdgpu_kernel void @kernel2() #0 {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar1)
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar1, i32 9)
    call void @llvm.amdgcn.s.barrier.wait(i16 1)

    call void @func2()
    ret void
}

; SOUT:      .amdhsa_kernel wgbarr_as_gv
; SOUT:        .amdhsa_named_barrier_count 0
define amdgpu_kernel void @wgbarr_as_gv() {
    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @wgbarr, i32 7)
    call void @llvm.amdgcn.s.barrier.wait(i16 -1)
    ret void
}

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }

!0 = !{i32 -1, i32 0}
