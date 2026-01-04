; RUN: not llc --global-isel %s -filetype=null 2>&1 | FileCheck %s
target triple = "spirv64"

define void @memcpy(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: G_MEMCPY %{{.*}}:pid(p9), %{{.*}}:pid(p0), %{{.*}}:iid(s64), 0
  call void @llvm.memcpy.p9.p0.i64(ptr addrspace(9) %a, ptr null, i32 1, i1 0)
  ret void
}
