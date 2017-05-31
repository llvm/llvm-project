; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}

; CHECK: !0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
; CHECK: !1 = !{i32 1, !"Objective-C Class Properties", i32 0}
