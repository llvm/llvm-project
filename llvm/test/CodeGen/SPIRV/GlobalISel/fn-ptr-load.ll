; RUN: not llc --global-isel %s -filetype=null 2>&1 | FileCheck %s
target triple = "spirv64"

define void @do_load(ptr addrspace(9) %a, ptr addrspace(1) %out) {
; CHECK: unable to legalize instruction: %{{.*}}:iid(s32) = G_LOAD %{{.*}}:pid(p9) 
  %val = load i32, ptr addrspace(9) %a
  store i32 %val, ptr addrspace(1) %out
  ret void
}
