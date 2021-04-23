; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@g = dso_local global i32 0, align 4

define i32 @test_load() sanitize_address {
entry:
  %tmp = load i32, i32* @g, align 4
  ret i32 %tmp
}

!llvm.module.flags = !{!0, !1}

;; Due to -fasynchronous-unwind-tables.
!0 = !{i32 7, !"uwtable", i32 1}

;; Due to -fno-omit-frame-pointer.
!1 = !{i32 7, !"frame-pointer", i32 2}

;; Set the uwtable attribute on ctor/dtor.
; CHECK: define internal void @asan.module_ctor() #[[#ATTR:]]
; CHECK: define internal void @asan.module_dtor() #[[#ATTR]]
; CHECK: attributes #[[#ATTR]] = { nounwind uwtable "frame-pointer"="all" }
