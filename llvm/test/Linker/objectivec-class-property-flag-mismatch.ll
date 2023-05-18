; RUN: llvm-as < %s -o %t1.bc
; RUN: llvm-as < %p/Inputs/objectivec-class-property-flag-mismatch.ll -o %t2.bc


; RUN: llvm-link %t1.bc %t2.bc -S | FileCheck %s
; RUN: llvm-link %t2.bc %t1.bc -S | FileCheck %s


; CHECK: !0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
; CHECK: !1 = !{i32 4, !"Objective-C Class Properties", i32 0}



!llvm.module.flags = !{!0}

!0 = !{i32 1, !"Objective-C Image Info Version", i32 0}
