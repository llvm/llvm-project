; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri < %s | FileCheck %s

; CHECK-LABEL: 'addrspacecast_global_to_flat'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast i8 addrspace(1)* %ptr to i8*
define i8* @addrspacecast_global_to_flat(i8 addrspace(1)* %ptr) #0 {
  %cast = addrspacecast i8 addrspace(1)* %ptr to i8*
  ret i8* %cast
}

; CHECK-LABEL: 'addrspacecast_global_to_flat_v2'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast <2 x i8 addrspace(1)*> %ptr to <2 x i8*>
define <2 x i8*> @addrspacecast_global_to_flat_v2(<2 x i8 addrspace(1)*> %ptr) #0 {
  %cast = addrspacecast <2 x i8 addrspace(1)*> %ptr to <2 x i8*>
  ret <2 x i8*> %cast
}

; CHECK-LABEL: 'addrspacecast_global_to_flat_v32'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast <32 x i8 addrspace(1)*> %ptr to <32 x i8*>
define <32 x i8*> @addrspacecast_global_to_flat_v32(<32 x i8 addrspace(1)*> %ptr) #0 {
  %cast = addrspacecast <32 x i8 addrspace(1)*> %ptr to <32 x i8*>
  ret <32 x i8*> %cast
}

; CHECK-LABEL: 'addrspacecast_local_to_flat'
; CHECK: estimated cost of 1 for {{.*}} addrspacecast i8 addrspace(3)* %ptr to i8*
define i8* @addrspacecast_local_to_flat(i8 addrspace(3)* %ptr) #0 {
  %cast = addrspacecast i8 addrspace(3)* %ptr to i8*
  ret i8* %cast
}

; CHECK-LABEL: 'addrspacecast_local_to_flat_v2'
; CHECK: estimated cost of 2 for {{.*}} addrspacecast <2 x i8 addrspace(3)*> %ptr to <2 x i8*>
define <2 x i8*> @addrspacecast_local_to_flat_v2(<2 x i8 addrspace(3)*> %ptr) #0 {
  %cast = addrspacecast <2 x i8 addrspace(3)*> %ptr to <2 x i8*>
  ret <2 x i8*> %cast
}

; CHECK-LABEL: 'addrspacecast_local_to_flat_v32'
; CHECK: estimated cost of 32 for {{.*}} addrspacecast <32 x i8 addrspace(3)*> %ptr to <32 x i8*>
define <32 x i8*> @addrspacecast_local_to_flat_v32(<32 x i8 addrspace(3)*> %ptr) #0 {
  %cast = addrspacecast <32 x i8 addrspace(3)*> %ptr to <32 x i8*>
  ret <32 x i8*> %cast
}

; CHECK-LABEL: 'addrspacecast_flat_to_local'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast i8* %ptr to i8 addrspace(3)*
define i8 addrspace(3)* @addrspacecast_flat_to_local(i8* %ptr) #0 {
  %cast = addrspacecast i8* %ptr to i8 addrspace(3)*
  ret i8 addrspace(3)* %cast
}

; CHECK-LABEL: 'addrspacecast_flat_to_local_v2'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast <2 x i8*> %ptr to <2 x i8 addrspace(3)*>
define <2 x i8 addrspace(3)*> @addrspacecast_flat_to_local_v2(<2 x i8*> %ptr) #0 {
  %cast = addrspacecast <2 x i8*> %ptr to <2 x i8 addrspace(3)*>
  ret <2 x i8 addrspace(3)*> %cast
}

; CHECK-LABEL: 'addrspacecast_flat_to_local_v32'
; CHECK: estimated cost of 0 for {{.*}} addrspacecast <32 x i8*> %ptr to <32 x i8 addrspace(3)*>
define <32 x i8 addrspace(3)*> @addrspacecast_flat_to_local_v32(<32 x i8*> %ptr) #0 {
  %cast = addrspacecast <32 x i8*> %ptr to <32 x i8 addrspace(3)*>
  ret <32 x i8 addrspace(3)*> %cast
}

attributes #0 = { nounwind readnone }
