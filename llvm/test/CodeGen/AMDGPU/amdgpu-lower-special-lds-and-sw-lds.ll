; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-special-lds,amdgpu-sw-lower-lds -amdgpu-asan-instrument-lds=false < %s 2>&1 | FileCheck %s
; RUN: llc < %s -enable-new-pm -stop-after=amdgpu-sw-lower-lds -amdgpu-asan-instrument-lds=false -mtriple=amdgcn-amd-amdhsa | FileCheck %s

%class.ExpAmdWorkgroupWaveBarrier = type { target("amdgcn.named.barrier", 0) }
@bar2 = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison
@bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@bar1 = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison
@lds1 = internal addrspace(3) global [1 x i8] poison, align 4

;.
; CHECK: @bar2 = internal addrspace(3) global [2 x target("amdgcn.named.barrier", 0)] poison, !absolute_symbol [[META0:![0-9]+]]
; CHECK: @bar3 = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison, !absolute_symbol [[META1:![0-9]+]]
; CHECK: @bar1 = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison, !absolute_symbol [[META2:![0-9]+]]
; CHECK: @bar1.barkernel = internal addrspace(3) global [4 x %class.ExpAmdWorkgroupWaveBarrier] poison, !absolute_symbol [[META2]]
;
define void @foo() #0 {
; CHECK-LABEL: define void @foo(
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar3, i32 7)
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar3)
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar3, i32 7)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar3)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  ret void
}

define void @bar() #0 {
; CHECK-LABEL: define void @bar(
; CHECK:    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar2, i32 7)
; CHECK:    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar2)
; CHECK:    call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar2, i32 7)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar2)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  store i8 7, ptr addrspace(3) @lds1, align 4
  ret void
}

define amdgpu_kernel void @fookernel() #0 {
; CHECK-LABEL: define amdgpu_kernel void @fookernel(
; CHECK:    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 11)
; CHECK:    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
; CHECK:    call void @llvm.amdgcn.s.barrier.wait(i16 1)
; CHECK:    [[STATE:%.*]] = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) @bar1)
; CHECK:    call void @llvm.amdgcn.s.barrier()
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 11)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  %state = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(3) @bar1)
  call void @llvm.amdgcn.s.barrier()
  call void @foo()
  call void @bar()
  store i8 9, ptr addrspace(3) @lds1, align 4
  ret void
}

define amdgpu_kernel void @barkernel() #0 {
; CHECK-LABEL: define amdgpu_kernel void @barkernel(
; CHECK:    call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1.barkernel, i32 9)
; CHECK:    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1.barkernel)
; CHECK:    call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar1, i32 9)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar1)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  call void @bar()
  store i8 10, ptr addrspace(3) @lds1, align 4
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

attributes #0 = { nounwind sanitize_address }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nosanitize_address", i32 1}
