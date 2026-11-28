; RUN: not llc -O2 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o /dev/null 2>&1 | FileCheck %s

; addrspacecast(inttoptr(4096)) followed by a constant GEP reaches G_PTR_ADD
; with an integer-typed base. Selection reports that base instead of replacing
; it with null.

; CHECK{LITERAL}: error: <unknown>:0:0: in function f void (): address space cast of a non-null integer used as a G_PTR_ADD base
; CHECK: LLVM ERROR: cannot select: {{.*}} = G_PTR_ADD {{.*}} (in function: f)

@p = addrspace(1) global ptr addrspace(4) getelementptr (
    i8, ptr addrspace(4) addrspacecast (
        ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1))
        to ptr addrspace(4)),
    i64 8), align 8

define spir_func void @f() addrspace(4) {
  unreachable
}
