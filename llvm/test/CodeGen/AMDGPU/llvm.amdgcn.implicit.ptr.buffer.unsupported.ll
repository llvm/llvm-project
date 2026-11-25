; Mesa shaders on targets with architected flat scratch don't have an implicit
; buffer pointer.
; RUN: not llc -global-isel=0 -mtriple=amdgpu11.00-mesa-mesa3d -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=SUBTARGET %s
; RUN: not llc -global-isel -mtriple=amdgpu11.00-mesa-mesa3d -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=SUBTARGET %s

; Only Mesa shaders have an implicit buffer pointer.
; RUN: not llc -global-isel=0 -mtriple=amdgpu10.30-mesa-mesa3d -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=CC %s
; RUN: not llc -global-isel -mtriple=amdgpu10.30-mesa-mesa3d -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=CC %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu10.30-- -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=OS %s
; RUN: not llc -global-isel -mtriple=amdgpu10.30-- -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=OS %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu10.30-amd-amdpal -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=OS %s
; RUN: not llc -global-isel -mtriple=amdgpu10.30-amd-amdpal -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=OS %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu7.00-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefixes=OS,HSA-SDAG %s
; RUN: not llc -global-isel -mtriple=amdgpu7.00-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=OS %s

; HSA-SDAG: error: {{.*}}in function test_ps{{.*}}: unsupported non-compute shaders with HSA
; SUBTARGET: error: {{.*}}in function test_ps{{.*}}: intrinsic not supported on subtarget
; OS: error: {{.*}}in function test_ps{{.*}}: intrinsic not supported for OS
define amdgpu_ps i32 @test_ps() {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  ret i32 %value
}

; HSA-SDAG: error: {{.*}}in function test_cs{{.*}}: unsupported non-compute shaders with HSA
; SUBTARGET: error: {{.*}}in function test_cs{{.*}}: intrinsic not supported on subtarget
; OS: error: {{.*}}in function test_cs{{.*}}: intrinsic not supported for OS
define amdgpu_cs i32 @test_cs() {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  ret i32 %value
}

; SUBTARGET: error: {{.*}}in function test_kernel{{.*}}: intrinsic not supported on subtarget
; OS: error: {{.*}}in function test_kernel{{.*}}: intrinsic not supported for OS
; CC: error: {{.*}}in function test_kernel{{.*}}: intrinsic not supported for calling convention
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out) {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; HSA-SDAG: error: {{.*}}in function test_gfx_func{{.*}}: unsupported non-compute shaders with HSA
; SUBTARGET: error: {{.*}}in function test_gfx_func{{.*}}: intrinsic not supported on subtarget
; OS: error: {{.*}}in function test_gfx_func{{.*}}: intrinsic not supported for OS
; CC: error: {{.*}}in function test_gfx_func{{.*}}: intrinsic not supported for calling convention
define amdgpu_gfx i32 @test_gfx_func() {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  ret i32 %value
}

; SUBTARGET: error: {{.*}}in function test_func{{.*}}: intrinsic not supported on subtarget
; OS: error: {{.*}}in function test_func{{.*}}: intrinsic not supported for OS
; CC: error: {{.*}}in function test_func{{.*}}: intrinsic not supported for calling convention
define i32 @test_func() {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  ret i32 %value
}
