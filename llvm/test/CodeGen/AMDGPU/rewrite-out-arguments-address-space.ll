; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-any-address-space-out-arguments -amdgpu-rewrite-out-arguments < %s | FileCheck %s

; CHECK: %void_one_out_non_private_arg_i32_1_use = type { i32 }
; CHECK: %bitcast_pointer_as1 = type { <4 x i32> }

; CHECK-LABEL: define private %void_one_out_non_private_arg_i32_1_use @void_one_out_non_private_arg_i32_1_use.body(i32 addrspace(1)* %val) #0 {
; CHECK-NEXT: ret %void_one_out_non_private_arg_i32_1_use zeroinitializer

; CHECK-LABEL: define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)* %0) #1 {
; CHECK-NEXT: %2 = call %void_one_out_non_private_arg_i32_1_use @void_one_out_non_private_arg_i32_1_use.body(i32 addrspace(1)* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_non_private_arg_i32_1_use %2, 0
; CHECK-NEXT: store i32 %3, i32 addrspace(1)* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)* %val) #0 {
  store i32 0, i32 addrspace(1)* %val
  ret void
}

; CHECK-LABEL: define private %bitcast_pointer_as1 @bitcast_pointer_as1.body(<3 x i32> addrspace(1)* %out) #0 {
; CHECK-NEXT: %load = load volatile <4 x i32>, <4 x i32> addrspace(1)* undef
; CHECK-NEXT: %bitcast = bitcast <3 x i32> addrspace(1)* %out to <4 x i32> addrspace(1)*
; CHECK-NEXT: %1 = insertvalue %bitcast_pointer_as1 undef, <4 x i32> %load, 0
; CHECK-NEXT: ret %bitcast_pointer_as1 %1

; CHECK-LABEL: define void @bitcast_pointer_as1(<3 x i32> addrspace(1)* %0) #1 {
; CHECK-NEXT: %2 = call %bitcast_pointer_as1 @bitcast_pointer_as1.body(<3 x i32> addrspace(1)* undef)
define void @bitcast_pointer_as1(<3 x i32> addrspace(1)* %out) #0 {
  %load = load volatile <4 x i32>, <4 x i32> addrspace(1)* undef
  %bitcast = bitcast <3 x i32> addrspace(1)* %out to <4 x i32> addrspace(1)*
  store <4 x i32> %load, <4 x i32> addrspace(1)* %bitcast
  ret void
}

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes #1 = { alwaysinline nounwind }
attributes #0 = { nounwind }
