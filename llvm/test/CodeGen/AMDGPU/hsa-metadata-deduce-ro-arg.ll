; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

; CHECK:      - Name:            test_ro_arg
; CHECK-NEXT:   SymbolName:      'test_ro_arg@kd'
; CHECK-NEXT:   Args:
; CHECK-NEXT: - Name:            in
; CHECK-NEXT:   TypeName:        'float*'
; CHECK-NEXT:   Size:            8
; CHECK-NEXT:   Align:           8
; CHECK-NEXT:   ValueKind:       GlobalBuffer
; CHECK-NEXT:   AddrSpaceQual:   Global
; CHECK-NEXT:   AccQual:         ReadOnly
; CHECK-NEXT:   IsConst:         true
; CHECK-NEXT:   IsRestrict:      true
; CHECK-NEXT: - Name:            out
; CHECK-NEXT:   TypeName:        'float*'
; CHECK-NEXT:   Size:            8
; CHECK-NEXT:   Align:           8
; CHECK-NEXT:   ValueKind:       GlobalBuffer
; CHECK-NEXT:   AddrSpaceQual:   Global
; CHECK-NEXT:   AccQual:         Default

define amdgpu_kernel void @test_ro_arg(float addrspace(1)* noalias readonly %in, float addrspace(1)* %out)
    !kernel_arg_addr_space !0 !kernel_arg_access_qual !1 !kernel_arg_type !2
    !kernel_arg_base_type !2 !kernel_arg_type_qual !3 {
  ret void
}

!0 = !{i32 1, i32 1}
!1 = !{!"none", !"none"}
!2 = !{!"float*", !"float*"}
!3 = !{!"const restrict", !""}
