; RUN: llvm-link %s %p/Inputs/wrong-addrspace-gv-declaration.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/wrong-addrspace-gv-declaration.ll %s -S | FileCheck %s

; The address space is declared incorrectly here, so an addrspacecast
; is needed to link.

@is_really_as1_gv = external global i32
@is_really_as1_gv_other_type = external global i32

; CHECK-LABEL: @foo(
; CHECK: %load0 = load volatile i32, ptr addrspacecast (ptr addrspace(1) @is_really_as1_gv to ptr), align 4
; CHECK: %load1 = load volatile i32, ptr addrspacecast (ptr addrspace(1) @is_really_as1_gv_other_type to ptr), align 4
define void @foo() {
  %load0 = load volatile i32, ptr @is_really_as1_gv, align 4
  %load1 = load volatile i32, ptr @is_really_as1_gv_other_type, align 4
  ret void
}
