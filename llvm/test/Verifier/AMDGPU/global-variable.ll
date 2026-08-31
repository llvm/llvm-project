; RUN: not llvm-as %s --disable-output 2>&1 | FileCheck %s

target triple = "amdgpu7.00-amd-amdhsa"

; A global variable is never allowed in addrspace(13) (VGPR). That address space
; is a view of one wave's vector registers, so it cannot provide the storage a
; global needs. Every other address space is left alone.

; CHECK: global variable on amdgpu must not be in addrspace(13)
; CHECK-NEXT: ptr addrspace(13) @gv.13
@gv.13 = addrspace(13) global i32 0, align 4

; A declaration has no initializer, so this covers the path that returns early.
; CHECK: global variable on amdgpu must not be in addrspace(13)
; CHECK-NEXT: ptr addrspace(13) @gv.13.extern
@gv.13.extern = external addrspace(13) global i32, align 4

; CHECK: global variable on amdgpu must not be in addrspace(13)
; CHECK-NEXT: ptr addrspace(13) @gv.13.const
@gv.13.const = addrspace(13) constant [4 x i32] zeroinitializer, align 4

; CHECK-NOT: global variable on amdgpu
@gv.0 = global i32 0, align 4
@gv.1 = addrspace(1) global i32 0, align 4
@gv.3 = addrspace(3) global i32 undef, align 4
@gv.4 = addrspace(4) constant i32 0, align 4
@gv.5 = addrspace(5) global i32 0, align 4
