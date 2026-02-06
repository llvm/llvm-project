; RUN: not llc --global-isel %s -filetype=null 2>&1 | FileCheck %s
target triple = "spirv64"

define void @memset(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: G_MEMSET %{{.*}}:pid(p9), %{{.*}}:iid(s8), %{{.*}}:iid(s64)
  call void @llvm.memset.p9.i32(ptr addrspace(9) %a, i8 0, i32 1, i1 0)
  ret void
}
