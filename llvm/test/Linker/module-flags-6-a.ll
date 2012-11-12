; RUN: not llvm-link %s %p/module-flags-6-b.ll -S -o - 2>&1 | FileCheck %s

; Test module flags error messages.

; CHECK: linking module flags 'foo': IDs have conflicting values

!0 = metadata !{ i32 1, metadata !"foo", i32 37 }

!llvm.module.flags = !{ !0 }
