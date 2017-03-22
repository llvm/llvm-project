; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}tgid_x:
; EG: MEM_RAT_CACHELESS STORE_RAW T1.X
define amdgpu_kernel void @tgid_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_y:
; EG: MEM_RAT_CACHELESS STORE_RAW T1.Y
define amdgpu_kernel void @tgid_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_z:
; EG: MEM_RAT_CACHELESS STORE_RAW T1.Z
define amdgpu_kernel void @tgid_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_x:
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X
define amdgpu_kernel void @tidig_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_y:
; EG: MEM_RAT_CACHELESS STORE_RAW T0.Y
define amdgpu_kernel void @tidig_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_z:
; EG: MEM_RAT_CACHELESS STORE_RAW T0.Z
define amdgpu_kernel void @tidig_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_implicit:
; 36 prepended implicit bytes + 4(out pointer) + 4*4 = 56
; EG: VTX_READ_32 {{T[0-9]+\.[XYZW]}}, {{T[0-9]+\.[XYZW]}}, 56
define amdgpu_kernel void @test_implicit(i32 addrspace(1)* %out) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(7)* @llvm.r600.implicitarg.ptr()
  %header.ptr = bitcast i8 addrspace(7)* %implicitarg.ptr to i32 addrspace(7)*
  %gep = getelementptr i32, i32 addrspace(7)* %header.ptr, i32 4
  %value = load i32, i32 addrspace(7)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_implicit_dyn:
; 36 prepended implicit bytes + 8(out pointer + in) = 44
; EG: VTX_READ_32 {{T[0-9]+\.[XYZW]}}, {{T[0-9]+\.[XYZW]}}, 44
define amdgpu_kernel void @test_implicit_dyn(i32 addrspace(1)* %out, i32 %in) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(7)* @llvm.r600.implicitarg.ptr()
  %header.ptr = bitcast i8 addrspace(7)* %implicitarg.ptr to i32 addrspace(7)*
  %gep = getelementptr i32, i32 addrspace(7)* %header.ptr, i32 %in
  %value = load i32, i32 addrspace(7)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

declare i8 addrspace(7)* @llvm.r600.implicitarg.ptr() #0

declare i32 @llvm.r600.read.tgid.x() #0
declare i32 @llvm.r600.read.tgid.y() #0
declare i32 @llvm.r600.read.tgid.z() #0

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0
declare i32 @llvm.r600.read.tidig.z() #0

attributes #0 = { readnone }
