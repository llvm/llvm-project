; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck %s

; CHECK:      - Name:            test_ro_arg
; CHECK:        Args:
; CHECK-NEXT: - Size:            8
; CHECK-NEXT:   Align:           8
; CHECK-NEXT:   ValueKind:       GlobalBuffer
; CHECK-NEXT:   ValueType:       F32
; CHECK-NEXT:   AccQual:         ReadOnly
; CHECK-NEXT:   AddrSpaceQual:   Global
; CHECK-NEXT:   IsConst:         true
; CHECK-NEXT:   IsRestrict:      true
; CHECK-NEXT:   TypeName:        'float*'

; CHECK-NEXT: - Size:            8
; CHECK-NEXT:   Align:           8
; CHECK-NEXT:   ValueKind:       GlobalBuffer
; CHECK-NEXT:   ValueType:       F32
; CHECK-NEXT:   AccQual:         Default
; CHECK-NEXT:   AddrSpaceQual:   Global
; CHECK-NEXT:   TypeName:        'float*'

define amdgpu_kernel void @test_ro_arg(float addrspace(1)* noalias readonly %in, float addrspace(1)* %out)
    !kernel_arg_addr_space !0 !kernel_arg_access_qual !1 !kernel_arg_type !2
    !kernel_arg_base_type !2 !kernel_arg_type_qual !3 {
  ret void
}

!0 = !{i32 1, i32 1}
!1 = !{!"none", !"none"}
!2 = !{!"float*", !"float*"}
!3 = !{!"const restrict", !""}

