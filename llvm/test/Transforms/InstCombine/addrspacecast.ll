; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-p1:32:32:32-p2:16:16:16-n8:16:32:64"

define i32* @combine_redundant_addrspacecast(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast(
; CHECK: addrspacecast i32 addrspace(1)* %x to i32*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(3)*
  %z = addrspacecast i32 addrspace(3)* %y to i32*
  ret i32* %z
}

define <4 x i32*> @combine_redundant_addrspacecast_vector(<4 x i32 addrspace(1)*> %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast_vector(
; CHECK: addrspacecast <4 x i32 addrspace(1)*> %x to <4 x i32*>
; CHECK-NEXT: ret
  %y = addrspacecast <4 x i32 addrspace(1)*> %x to <4 x i32 addrspace(3)*>
  %z = addrspacecast <4 x i32 addrspace(3)*> %y to <4 x i32*>
  ret <4 x i32*> %z
}

define float* @combine_redundant_addrspacecast_types(i32 addrspace(1)* %x) nounwind {
; CHECK-LABEL: @combine_redundant_addrspacecast_types(
; CHECK: addrspacecast i32 addrspace(1)* %x to float*
; CHECK-NEXT: ret
  %y = addrspacecast i32 addrspace(1)* %x to i32 addrspace(3)*
  %z = addrspacecast i32 addrspace(3)* %y to float*
  ret float* %z
}

