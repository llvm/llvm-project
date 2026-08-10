; RUN: opt -strip-named-metadata < %s -S -o - | FileCheck %s

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.debugify = !{!0}

; CHECK-NOT: llvm.module.flags
; CHECK-NOT: Debug Info Version
; CHECK-NOT: llvm.debugify
