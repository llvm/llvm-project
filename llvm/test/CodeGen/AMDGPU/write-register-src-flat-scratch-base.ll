; RUN: llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s

; src_flat_scratch_base{,_lo,_hi} are isConstant register aliases, so writes to
; them are dead and get silently eliminated rather than lowered to a copy.

declare void @llvm.write_register.i32(metadata, i32) #0
declare void @llvm.write_register.i64(metadata, i64) #0

; CHECK-LABEL: {{^}}test_write_src_flat_scratch_base:{{.*$}}
; CHECK-NOT: src_flat_scratch_base
; CHECK: s_endpgm
define amdgpu_kernel void @test_write_src_flat_scratch_base(i64 %val) #0 {
  call void @llvm.write_register.i64(metadata !0, i64 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_src_flat_scratch_base_lo:{{.*$}}
; CHECK-NOT: src_flat_scratch_base
; CHECK: s_endpgm
define amdgpu_kernel void @test_write_src_flat_scratch_base_lo(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !1, i32 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_src_flat_scratch_base_hi:{{.*$}}
; CHECK-NOT: src_flat_scratch_base
; CHECK: s_endpgm
define amdgpu_kernel void @test_write_src_flat_scratch_base_hi(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !2, i32 %val)
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"src_flat_scratch_base"}
!1 = !{!"src_flat_scratch_base_lo"}
!2 = !{!"src_flat_scratch_base_hi"}
