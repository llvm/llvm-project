; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "A0"
; CHECK: target datalayout = "A0"


; CHECK: %alloca_scalar_no_align = alloca i32, align 4
; CHECK-NEXT: %alloca_scalar_align4 = alloca i32, align 4
; CHECK-NEXT: %alloca_scalar_no_align_metadata = alloca i32, align 4, !foo !0
; CHECK-NEXT: %alloca_scalar_align4_metadata = alloca i32, align 4, !foo !0
; CHECK-NEXT: %alloca_inalloca_scalar_no_align = alloca inalloca i32, align 4
; CHECK-NEXT: %alloca_inalloca_scalar_align4_metadata = alloca inalloca i32, align 4, !foo !0
define void @use_alloca() {
  %alloca_scalar_no_align = alloca i32, addrspace(0)
  %alloca_scalar_align4 = alloca i32, align 4, addrspace(0)
  %alloca_scalar_no_align_metadata = alloca i32, addrspace(0), !foo !0
  %alloca_scalar_align4_metadata = alloca i32, align 4, addrspace(0), !foo !0
  %alloca_inalloca_scalar_no_align = alloca inalloca i32, addrspace(0)
  %alloca_inalloca_scalar_align4_metadata = alloca inalloca i32, align 4, addrspace(0), !foo !0

  ret void
}

!0 = !{}
