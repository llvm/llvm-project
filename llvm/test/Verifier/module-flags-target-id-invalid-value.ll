; RUN: not opt -verify %s 2>&1 | FileCheck %s

; CHECK: invalid value for 'MergeTargetID' module flag (expected string)

!llvm.module.flags = !{ !0 }

!0 = !{ i32 8, !"target-id", i32 37 }
