; RUN: opt -O1 -S -o - -mtriple=amdgcn %s | FileCheck %s

; CHECK-LABEL: promote_pointer_kernargs
; CHECK-NEXT: addrspacecast i32* %{{.*}} to i32 addrspace(1)*
; CHECK-NEXT: addrspacecast i32* %{{.*}} to i32 addrspace(1)*
; CHECK-NEXT: load i32, i32 addrspace(1)*
; CHECK-NEXT: store i32 %{{.*}}, i32 addrspace(1)*
; CHECK-NEXT: ret void
define amdgpu_kernel void @promote_pointer_kernargs(i32* %out, i32* %in) {
  %v = load i32, i32* %in
  store i32 %v, i32* %out
  ret void
}
