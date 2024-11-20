;; Check that asan_globals is marked large under x86-64 medium code model.
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=asan -S | FileCheck %s --check-prefixes=CHECK,X8664
; RUN: opt < %s -mtriple=ppc64-unknown-linux-gnu -passes=asan -S | FileCheck %s --check-prefixes=CHECK,PPC

; CHECK: @__asan_global_global =
; X8664-SAME: code_model "large"
; PPC-NOT: code_model "large"

@global = global i32 0, align 4

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"Code Model", i32 3}