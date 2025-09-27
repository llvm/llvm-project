; RUN: opt -S -mtriple=amdgcn -mcpu=gfx1100 -passes=amdgpu-convert-wave-size < %s | FileCheck %s

define amdgpu_kernel void @test_not_wave32(ptr addrspace(1) %out) #0 {
  ; CHECK:  @test_not_wave32{{.*}}) #0
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 2
  %tmp = load i32, ptr addrspace(1) %gep
  store i32 %tmp, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @intr_non_convergent(ptr addrspace(1) nocapture %arg) #1 {
  ; CHECK: @intr_non_convergent{{.*}} #0
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize()
  %tmp1 = icmp ugt i32 %tmp, 32
  %tmp2 = select i1 %tmp1, i32 2, i32 1
  store i32 %tmp2, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @intr_convergent(ptr addrspace(1) nocapture %arg, i32 %X) #1 {
  ; CHECK: @intr_convergent{{.*}}) #1
bb:
  %tmp = icmp ugt i32 %X, 32
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %tmp)
  store i32 %ballot, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_barrier(ptr addrspace(1) %in, ptr addrspace(1) %out) #1 {
  ; CHECK: @test_barrier{{.*}}) #0
entry:
  %val = load <2 x half>, ptr addrspace(1) %in
  call void @llvm.amdgcn.s.barrier() #2
  store <2 x half> %val, ptr addrspace(1) %out
  ret void
}


define amdgpu_kernel void @test_read_exec(ptr addrspace(1) %out) #1 {
  ; CHECK: @test_read_exec{{.*}}) #1
  %exec = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %exec, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_read_vcc_lo(ptr addrspace(1) %out) #1 {
  ; CHECK: @test_read_vcc_lo{{.*}}) #1
  %vcc_lo = call i32 @llvm.read_register.i32(metadata !1)
  store i32 %vcc_lo, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_read_vcc_hi(ptr addrspace(1) %out) #1 {
  ; CHECK: @test_read_vcc_hi{{.*}}) #1
  %vcc_hi = call i32 @llvm.read_register.i32(metadata !2)
  store i32 %vcc_hi, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_lds_access(ptr addrspace(3) %out) #1 {
  ; CHECK: @test_lds_access{{.*}}) #1
  %gep = getelementptr i32, ptr addrspace(3) %out, i32 2
  %tmp = load i32, ptr addrspace(3) %gep
  store i32 %tmp, ptr addrspace(3) %out
  ret void
}

define amdgpu_kernel void @test_addrspacecast_to_lds(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 16
  %ptr = addrspacecast ptr addrspace(1) %gep to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_bitcast_to_lds_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 16
  %lds = inttoptr i32 0 to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %lds
  store i32 %val, ptr addrspace(1) %out
  ret void
}

@lds = addrspace(3) global [256 x i32] zeroinitializer

define amdgpu_kernel void @test_use_global_lds_object(ptr addrspace(1) %out, i1 %p) #0 {
  %gep = getelementptr [256 x i32], ptr addrspace(3) @lds, i32 0, i32 10
  %ld = load i32, ptr addrspace(3) %gep
  store i32 %ld, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_simple_loop(ptr addrspace(1) nocapture %arg) #1 {
  ; CHECK: @test_simple_loop{{.*}}) #1
bb:
  br label %bb2

bb1:
  ret void

bb2:
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb2 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

define amdgpu_kernel void @test_nested_loop(ptr addrspace(1) nocapture %arg) #1 {
  ; CHECK: @test_nested_loop{{.*}}) #1
bb:
  br label %bb2

bb1:
  ret void

bb2:
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb4 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 8
  br label %bb3

bb3:
  %tmp4 = phi i32 [ 0, %bb2 ], [ %tmp5, %bb3 ]
  %tmp5 = add nuw nsw i32 %tmp4, 1
  %tmp6 = icmp eq i32 %tmp5, 128
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp6, label %bb4, label %bb3

bb4:
  br i1 %tmp3, label %bb1, label %bb2
}

declare void @llvm.amdgcn.s.sleep(i32)
declare i32 @llvm.amdgcn.wavefrontsize()
declare i32 @llvm.amdgcn.ballot.i32(i1)
declare i32 @llvm.read_register.i32(metadata)
declare i64 @llvm.read_register.i64(metadata)

attributes #0 = { nounwind "target-features"="+wavefrontsize64" }
attributes #1 = { nounwind "target-features"="+wavefrontsize32" }

!0 = !{!"exec"}
!1 = !{!"vcc_lo"}
!2 = !{!"vcc_hi"}
