; RUN: opt -run-twice -passes=verify -S -o - %s | FileCheck %s

; This test is used to check metadata attached to global variable declarations
; are copied when CloneModule(). This is required by out-of-tree passes.

; CHECK: @g = external addrspace(64) global i32, !spirv.InOut !0

@g = external addrspace(64) global i32, !spirv.InOut !0

!0 = !{i32 1}
