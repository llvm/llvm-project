;; Verify that the synthetic functions inherit their flags from the corresponding
;; BTE and return address signing module flags.
; RUN: opt < %s -passes=asan -S | FileCheck %s
; REQUIRES: aarch64-registered-target

target triple = "aarch64-unknown-linux-gnu"

@g = dso_local global i32 0, align 4

define i32 @test_load() sanitize_memory {
entry:
  %tmp = load i32, ptr @g, align 4
  ret i32 %tmp
}

!llvm.module.flags = !{!0, !1}

;; Due to -fasynchronous-unwind-tables.
!0 = !{i32 7, !"uwtable", i32 2}

;; Due to -fno-omit-frame-pointer.
!1 = !{i32 7, !"frame-pointer", i32 2}

!llvm.module.flags = !{!2, !3, !4}

!2 = !{i32 8, !"branch-target-enforcement", i32 1}
!3 = !{i32 8, !"sign-return-address", i32 1}
!4 = !{i32 8, !"sign-return-address-all", i32 0}

;; Set the uwtable attribute on ctor/dtor.
; CHECK: define internal void @asan.module_ctor() #[[#ATTR:]]
; CHECK: define internal void @asan.module_dtor() #[[#ATTR]]
; CHECK: attributes #[[#ATTR]] = { nounwind uwtable "branch-target-enforcement" "frame-pointer"="all" "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" }
