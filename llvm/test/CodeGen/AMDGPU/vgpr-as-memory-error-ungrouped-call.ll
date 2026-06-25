; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; A file-using device function that is not reached from any kernel forms a group
; on its own; a call to a defined function outside that group would clobber the
; file's reserved registers, so it is diagnosed (not just external/indirect
; calls).

@g = internal addrspace(13) global i32 poison

define void @other() {
  ret void
}

; CHECK: error: {{.*}}'VGPR as memory' is not supported in a function that makes an indirect call or a call outside its call graph
define void @dev_user() {
  store i32 1, ptr addrspace(13) @g
  call void @other()
  ret void
}
