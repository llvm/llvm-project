;; Verify that the synthetic functions get "hw-shadow-stack" from the module
;; flag of the same name.
; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: sed 's/"hw-shadow-stack", i32 1/"hw-shadow-stack", i32 0/' %s \
; RUN:   | opt -passes=asan -S | FileCheck --check-prefix=NOSHADOWSTACK %s
; REQUIRES: riscv-registered-target

target triple = "riscv64-unknown-linux-gnu"

@g = dso_local global i32 0, align 4

define i32 @test_load() sanitize_address {
entry:
  %tmp = load i32, ptr @g, align 4
  ret i32 %tmp
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"hw-shadow-stack", i32 1}

; CHECK: define internal void @asan.module_ctor() #[[#ATTR:]]
; CHECK: define internal void @asan.module_dtor() #[[#ATTR]]
; CHECK: attributes #[[#ATTR]] = { nounwind "hw-shadow-stack" }

; NOSHADOWSTACK-NOT: attributes {{.*}}"hw-shadow-stack"
