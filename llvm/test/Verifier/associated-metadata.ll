; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: associated value must be pointer typed
; CHECK-NEXT: ptr addrspace(1) @associated.int
; CHECK-NEXT: !0 = !{i32 1}
@associated.int = external addrspace(1) constant [8 x i8], !associated !0

; CHECK: associated value must be pointer typed
; CHECK-NEXT: ptr addrspace(1) @associated.float
; CHECK-NEXT: !1 = !{float 1.000000e+00}
@associated.float = external addrspace(1) constant [8 x i8], !associated !1

; CHECK: associated metadata must have one operand
; CHECK-NEXT: ptr addrspace(1) @associated.too.many.ops
; CHECK-NEXT: !2 = !{ptr @gv.decl0, ptr @gv.decl1}
@associated.too.many.ops = external addrspace(1) constant [8 x i8], !associated !2

; CHECK: associated metadata must have one operand
; CHECK-NEXT: ptr addrspace(1) @associated.empty
; CHECK-NEXT: !3 = !{}
@associated.empty = external addrspace(1) constant [8 x i8], !associated !3

; CHECK: associated metadata must have a global value
; CHECK-NEXT: ptr addrspace(1) @associated.null.metadata
; CHECK-NEXT: !4 = !{null}
@associated.null.metadata = external addrspace(1) constant [8 x i8], !associated !4

; CHECK: global values should not associate to themselves
; CHECK-NEXT: ptr @associated.self
; CHECK-NEXT: !5 = !{ptr @associated.self}
@associated.self = external constant [8 x i8], !associated !5

; CHECK: associated metadata must be ValueAsMetadata
; CHECK-NEXT: ptr @associated.string
; CHECK-NEXT: !6 = !{!"string"}
@associated.string = external constant [8 x i8], !associated !6

@gv.decl0 = external constant [8 x i8]
@gv.decl1 = external constant [8 x i8]

!0 = !{i32 1}
!1 = !{float 1.000000e+00}
!2 = !{ptr @gv.decl0, ptr @gv.decl1}
!3 = !{}
!4 = !{null}
!5 = !{ptr @associated.self}
!6 = !{!"string"}
