; RUN: opt -module-summary < %s | llvm-dis | FileCheck %s

@__oclc_ABI_version = linkonce_odr hidden addrspace(4) constant i32 500, align 4
@_ZL20__oclc_ABI_version__ = internal alias i32, addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr)

; CHECK: ^1 = gv: (name: "__oclc_ABI_version", summaries: (variable: (module: ^0, flags: {{.*}})))
; CHECK: ^2 = gv: (name: "_ZL20__oclc_ABI_version__", summaries: (alias: (module: ^0, flags: {{.*}}, aliasee: ^1)))
