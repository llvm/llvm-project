; RUN: not llc --global-isel %s -filetype=null 2>&1 | FileCheck %s
target triple = "spirv64"

define void @addrspacecast(ptr addrspace(9) %a) {
; CHECK: unable to legalize instruction: %{{.*}}:pid(p4) = G_ADDRSPACE_CAST %{{.*}}:pid(p9)
  %res1 = addrspacecast ptr addrspace(9) %a to ptr addrspace(4)
  store i8 0, ptr addrspace(4) %res1
  ret void
}
