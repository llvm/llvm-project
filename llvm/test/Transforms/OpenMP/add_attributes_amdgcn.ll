; RUN: opt < %s -S -openmpopt        | FileCheck %s
; RUN: opt < %s -S -passes=openmpopt | FileCheck %s
; RUN: opt < %s -S -openmpopt        -openmp-ir-builder-optimistic-attributes | FileCheck %s --check-prefix=OPTIMISTIC
; RUN: opt < %s -S -passes=openmpopt -openmp-ir-builder-optimistic-attributes | FileCheck %s --check-prefix=OPTIMISTIC

target triple = "amdgcn-amd-amdhsa"

define void @call_all(i64 %arg) {
  call void @__kmpc_syncwarp(i64 %arg)
  call i64 @__kmpc_warp_active_thread_mask()
  ret void
}

declare i64 @__kmpc_warp_active_thread_mask()

declare void @__kmpc_syncwarp(i64)

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare i64 @__kmpc_warp_active_thread_mask()

; CHECK: ; Function Attrs: convergent nounwind
; CHECK-NEXT: declare void @__kmpc_syncwarp(i64)

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare i64 @__kmpc_warp_active_thread_mask()

; OPTIMISTIC: ; Function Attrs: convergent nounwind
; OPTIMISTIC-NEXT: declare void @__kmpc_syncwarp(i64)
