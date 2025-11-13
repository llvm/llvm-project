; RUN: not llc --global-isel %s -o /dev/null 2>&1 | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1-P9-A0"
target triple = "spirv64-intel"

define void @memcpy(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: G_MEMCPY %{{.*}}:pid(p9), %{{.*}}:pid(p0), %{{.*}}:iid(s64), 0
  call void @llvm.memcpy.p9.p0.i64(ptr addrspace(9) %a, ptr null, i32 1, i1 0)
  ret void
}
