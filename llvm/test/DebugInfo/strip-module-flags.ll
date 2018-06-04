; RUN: opt -strip-module-flags < %s -S -o - | FileCheck %s

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK-NOT: llvm.module.flags
; CHECK-NOT: Debug Info Version
