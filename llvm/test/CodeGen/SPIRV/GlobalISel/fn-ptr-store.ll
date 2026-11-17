; RUN: not llc --global-isel %s -filetype=null 2>&1 | FileCheck %s
target triple = "spirv64"

define void @do_store(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: G_STORE %{{.*}}:iid(s32), %{{.*}}:pid(p9) 
  store i32 5, ptr addrspace(9) %a
  ret void
}
