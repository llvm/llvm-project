; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s


; CHECK: .visible .global .align 4 .u32 device_g;
@device_g = addrspace(1) global i32 zeroinitializer
; CHECK: .visible .global .attribute(.managed) .align 4 .u32 managed_g;
@managed_g = addrspace(1) global i32 zeroinitializer

; CHECK: .extern .global .align 4 .u32 decl_g;
@decl_g = external addrspace(1) global i32, align 4
; CHECK: .extern .global .attribute(.managed) .align 8 .b32 managed_decl_g;
@managed_decl_g = external addrspace(1) global i32*, align 8

!nvvm.annotations = !{!0, !1}
!0 = !{i32 addrspace(1)* @managed_g, !"managed", i32 1}
!1 = !{i32* addrspace(1)* @managed_decl_g, !"managed", i32 1}
