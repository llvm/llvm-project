; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=HSA -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefix=MESA -check-prefix=ALL %s

; ALL-LABEL: {{^}}test:
; HSA: enable_sgpr_kernarg_segment_ptr = 1
; HSA: s_load_dword s{{[0-9]+}}, s[4:5], 0xa

; MESA: s_load_dword s{{[0-9]+}}, s[0:1], 0xa
define void @test(i32 addrspace(1)* %out) #1 {
  %kernarg.segment.ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr()
  %header.ptr = bitcast i8 addrspace(2)* %kernarg.segment.ptr to i32 addrspace(2)*
  %gep = getelementptr i32, i32 addrspace(2)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(2)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit:
; 10 + 9 (36 prepended implicit bytes) + 2(out pointer) = 21 = 0x15
; MESA: s_load_dword s{{[0-9]+}}, s[0:1], 0x15
define void @test_implicit(i32 addrspace(1)* %out) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr()
  %header.ptr = bitcast i8 addrspace(2)* %implicitarg.ptr to i32 addrspace(2)*
  %gep = getelementptr i32, i32 addrspace(2)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(2)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

declare i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i8 addrspace(2)* @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
