; RUN: llc -O0 < %s -march=nvptx -mcpu=sm_20 | FileCheck %s -check-prefix=PTX32
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s -check-prefix=PTX64

define i32 @conv1(i32 addrspace(1)* %ptr) {
; PTX32: conv1
; PTX32: cvta.global.u32
; PTX32: ld.u32
; PTX64: conv1
; PTX64: cvta.global.u64
; PTX64: ld.u32
  %genptr = addrspacecast i32 addrspace(1)* %ptr to i32*
  %val = load i32, i32* %genptr
  ret i32 %val
}

define i32 @conv2(i32 addrspace(3)* %ptr) {
; PTX32: conv2
; PTX32: cvta.shared.u32
; PTX32: ld.u32
; PTX64: conv2
; PTX64: cvta.shared.u64
; PTX64: ld.u32
  %genptr = addrspacecast i32 addrspace(3)* %ptr to i32*
  %val = load i32, i32* %genptr
  ret i32 %val
}

define i32 @conv3(i32 addrspace(4)* %ptr) {
; PTX32: conv3
; PTX32: cvta.const.u32
; PTX32: ld.u32
; PTX64: conv3
; PTX64: cvta.const.u64
; PTX64: ld.u32
  %genptr = addrspacecast i32 addrspace(4)* %ptr to i32*
  %val = load i32, i32* %genptr
  ret i32 %val
}

define i32 @conv4(i32 addrspace(5)* %ptr) {
; PTX32: conv4
; PTX32: cvta.local.u32
; PTX32: ld.u32
; PTX64: conv4
; PTX64: cvta.local.u64
; PTX64: ld.u32
  %genptr = addrspacecast i32 addrspace(5)* %ptr to i32*
  %val = load i32, i32* %genptr
  ret i32 %val
}

define i32 @conv5(i32* %ptr) {
; PTX32: conv5
; PTX32: cvta.to.global.u32
; PTX32: ld.global.u32
; PTX64: conv5
; PTX64: cvta.to.global.u64
; PTX64: ld.global.u32
  %specptr = addrspacecast i32* %ptr to i32 addrspace(1)*
  %val = load i32, i32 addrspace(1)* %specptr
  ret i32 %val
}

define i32 @conv6(i32* %ptr) {
; PTX32: conv6
; PTX32: cvta.to.shared.u32
; PTX32: ld.shared.u32
; PTX64: conv6
; PTX64: cvta.to.shared.u64
; PTX64: ld.shared.u32
  %specptr = addrspacecast i32* %ptr to i32 addrspace(3)*
  %val = load i32, i32 addrspace(3)* %specptr
  ret i32 %val
}

define i32 @conv7(i32* %ptr) {
; PTX32: conv7
; PTX32: cvta.to.const.u32
; PTX32: ld.const.u32
; PTX64: conv7
; PTX64: cvta.to.const.u64
; PTX64: ld.const.u32
  %specptr = addrspacecast i32* %ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %specptr
  ret i32 %val
}

define i32 @conv8(i32* %ptr) {
; PTX32: conv8
; PTX32: cvta.to.local.u32
; PTX32: ld.local.u32
; PTX64: conv8
; PTX64: cvta.to.local.u64
; PTX64: ld.local.u32
  %specptr = addrspacecast i32* %ptr to i32 addrspace(5)*
  %val = load i32, i32 addrspace(5)* %specptr
  ret i32 %val
}
