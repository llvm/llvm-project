; RUN: not llc --global-isel %s -o /dev/null 2>&1 | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1-P9-A0"
target triple = "spirv64-intel"

define void @addrspacecast(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: %{{.*}}:pid(p4) = G_ADDRSPACE_CAST %{{.*}}:pid(p9)
  %res1 = addrspacecast ptr addrspace(9) %a to ptr addrspace(4)
  ret void
}
