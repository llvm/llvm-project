; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

; Test that global aliases are allowed to be constant addrspacecast

@i = internal addrspace(1) global i8 42
@ia = internal alias i8 addrspace(2)*, addrspacecast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(3)*)
; CHECK: @ia = internal alias i8 addrspace(2)*, addrspacecast (i8 addrspace(2)* addrspace(1)* bitcast (i8 addrspace(1)* @i to i8 addrspace(2)* addrspace(1)*) to i8 addrspace(2)* addrspace(3)*)
