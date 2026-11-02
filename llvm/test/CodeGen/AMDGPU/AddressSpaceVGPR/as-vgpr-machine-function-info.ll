; RUN: llc -global-isel=0 -mtriple=amdgpu12.00-- -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.00-- -stop-after=instruction-select < %s | FileCheck %s

; Lowering an access to address space 13 records it in the function info,
; which AMDGPULowerVGPREncoding checks before looking for the indexed pseudos.

; CHECK-LABEL: name: with_access
; CHECK: hasVGPRAsMemoryAccess: true
define i32 @with_access(ptr addrspace(13) %p) {
  %v = load i32, ptr addrspace(13) %p
  ret i32 %v
}

; CHECK-LABEL: name: without_access
; CHECK: hasVGPRAsMemoryAccess: false
define i32 @without_access(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  ret i32 %v
}
