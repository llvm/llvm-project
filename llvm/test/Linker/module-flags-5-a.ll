; RUN: not llvm-link %s %p/module-flags-5-b.ll -S -o - 2>&1 | FileCheck %s

; Test the 'override' error.

; CHECK: linking module flags 'foo': IDs have conflicting override values in '{{.*}}module-flags-5-b.ll' and 'llvm-link'

!0 = !{ i32 4, !"foo", i32 927 }

!llvm.module.flags = !{ !0 }
