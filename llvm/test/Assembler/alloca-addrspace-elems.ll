; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "A5"
; CHECK: target datalayout = "A5"


; CHECK: %alloca_array_no_align = alloca i32, i32 9, align 4, addrspace(5)
; CHECK-NEXT: %alloca_array_align4 = alloca i32, i32 9, align 4, addrspace(5)
; CHECK-NEXT: %alloca_array_no_align_metadata = alloca i32, i32 9, align 4, addrspace(5), !foo !0
; CHECK-NEXT: %alloca_array_align4_metadata = alloca i32, i32 9, align 4, addrspace(5), !foo !0
; CHECK-NEXT: %alloca_inalloca_array_no_align = alloca inalloca i32, i32 9, align 4, addrspace(5)
; CHECK-NEXT: %alloca_inalloca_array_align4_metadata = alloca inalloca i32, i32 9, align 4, addrspace(5), !foo !0

define void @use_alloca() {
  %alloca_array_no_align = alloca i32, i32 9, addrspace(5)
  %alloca_array_align4 = alloca i32, i32 9, align 4, addrspace(5)
  %alloca_array_no_align_metadata = alloca i32, i32 9, addrspace(5), !foo !0
  %alloca_array_align4_metadata = alloca i32, i32 9, align 4, addrspace(5), !foo !0
  %alloca_inalloca_array_no_align = alloca inalloca i32, i32 9, addrspace(5)
  %alloca_inalloca_array_align4_metadata = alloca inalloca i32, i32 9, align 4, addrspace(5), !foo !0

  ret void
}

!0 = !{}
