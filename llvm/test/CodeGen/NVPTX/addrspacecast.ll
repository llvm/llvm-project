; RUN: llc -O0 < %s -march=nvptx -mcpu=sm_20 | FileCheck %s -check-prefixes=ALL,CLS32,G32
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s -check-prefixes=ALL,NOPTRCONV,CLS64,G64
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_20 --nvptx-short-ptr| FileCheck %s -check-prefixes=ALL,PTRCONV,CLS64,G64
; RUN: %if ptxas && !ptxas-12.0 %{ llc -O0 < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -march=nvptx64 -mcpu=sm_20 --nvptx-short-ptr | %ptxas-verify %}

; ALL-LABEL: conv1
define i32 @conv1(ptr addrspace(1) %ptr) {
; G32: cvta.global.u32
; ALL-NOT: cvt.u64.u32
; G64: cvta.global.u64
; ALL: ld.u32
  %genptr = addrspacecast ptr addrspace(1) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv2
define i32 @conv2(ptr addrspace(3) %ptr) {
; CLS32: cvta.shared.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.shared.u64
; ALL: ld.u32
  %genptr = addrspacecast ptr addrspace(3) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv3
define i32 @conv3(ptr addrspace(4) %ptr) {
; CLS32: cvta.const.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.const.u64
; ALL: ld.u32
  %genptr = addrspacecast ptr addrspace(4) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv4
define i32 @conv4(ptr addrspace(5) %ptr) {
; CLS32: cvta.local.u32
; PTRCONV: cvt.u64.u32
; NOPTRCONV-NOT: cvt.u64.u32
; CLS64: cvta.local.u64
; ALL: ld.u32
  %genptr = addrspacecast ptr addrspace(5) %ptr to ptr
  %val = load i32, ptr %genptr
  ret i32 %val
}

; ALL-LABEL: conv5
define i32 @conv5(ptr %ptr) {
; CLS32: cvta.to.global.u32
; ALL-NOT: cvt.u64.u32
; CLS64: cvta.to.global.u64
; ALL: ld.global.u32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(1)
  %val = load i32, ptr addrspace(1) %specptr
  ret i32 %val
}

; ALL-LABEL: conv6
define i32 @conv6(ptr %ptr) {
; CLS32: cvta.to.shared.u32
; CLS64: cvta.to.shared.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.shared.u32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(3)
  %val = load i32, ptr addrspace(3) %specptr
  ret i32 %val
}

; ALL-LABEL: conv7
define i32 @conv7(ptr %ptr) {
; CLS32: cvta.to.const.u32
; CLS64: cvta.to.const.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.const.u32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(4)
  %val = load i32, ptr addrspace(4) %specptr
  ret i32 %val
}

; ALL-LABEL: conv8
define i32 @conv8(ptr %ptr) {
; CLS32: cvta.to.local.u32
; CLS64: cvta.to.local.u64
; PTRCONV: cvt.u32.u64
; NOPTRCONV-NOT: cvt.u32.u64
; ALL: ld.local.u32
  %specptr = addrspacecast ptr %ptr to ptr addrspace(5)
  %val = load i32, ptr addrspace(5) %specptr
  ret i32 %val
}
