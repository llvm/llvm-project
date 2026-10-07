; RUN: not llc -mtriple=amdgpu12.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck %s

; src_flat_scratch_base{,_lo,_hi} require globally-addressable-scratch (gfx1250+),
; so named-register access to them should be rejected on earlier subtargets.

declare i32 @llvm.read_register.i32(metadata) #0
declare i64 @llvm.read_register.i64(metadata) #0
declare void @llvm.write_register.i32(metadata, i32) #0
declare void @llvm.write_register.i64(metadata, i64) #0

; CHECK: error: invalid register "src_flat_scratch_base" for subtarget.
define amdgpu_kernel void @test_read_src_flat_scratch_base(ptr addrspace(1) %out) nounwind {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: invalid register "src_flat_scratch_base_lo" for subtarget.
define amdgpu_kernel void @test_read_src_flat_scratch_base_lo(ptr addrspace(1) %out) nounwind {
  %v = call i32 @llvm.read_register.i32(metadata !1)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: invalid register "src_flat_scratch_base_hi" for subtarget.
define amdgpu_kernel void @test_read_src_flat_scratch_base_hi(ptr addrspace(1) %out) nounwind {
  %v = call i32 @llvm.read_register.i32(metadata !2)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: invalid register "src_flat_scratch_base" for subtarget.
define amdgpu_kernel void @test_write_src_flat_scratch_base(i64 %val) nounwind {
  call void @llvm.write_register.i64(metadata !0, i64 %val)
  ret void
}

; CHECK: error: invalid register "src_flat_scratch_base_lo" for subtarget.
define amdgpu_kernel void @test_write_src_flat_scratch_base_lo(i32 %val) nounwind {
  call void @llvm.write_register.i32(metadata !1, i32 %val)
  ret void
}

; CHECK: error: invalid register "src_flat_scratch_base_hi" for subtarget.
define amdgpu_kernel void @test_write_src_flat_scratch_base_hi(i32 %val) nounwind {
  call void @llvm.write_register.i32(metadata !2, i32 %val)
  ret void
}

!0 = !{!"src_flat_scratch_base"}
!1 = !{!"src_flat_scratch_base_lo"}
!2 = !{!"src_flat_scratch_base_hi"}
