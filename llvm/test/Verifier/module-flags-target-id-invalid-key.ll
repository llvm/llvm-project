; RUN: not opt -verify %s 2>&1 | FileCheck %s

; CHECK: Invalid key for 'MergeTargetID' module flag (expected 'target-id')

!llvm.module.flags = !{ !0 }

!0 = !{ i32 8, !"foo", !"amdgcn-amd-amdhsa-gfx908" }
