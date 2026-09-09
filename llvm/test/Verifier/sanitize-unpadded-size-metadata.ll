; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: sanitize.unpadded.size metadata must have one operand
; CHECK-NEXT: ptr @too.many.ops
; CHECK-NEXT: !0 = !{i64 4, i64 8}
@too.many.ops = global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !0

; CHECK: sanitize.unpadded.size metadata must have one operand
; CHECK-NEXT: ptr @empty
; CHECK-NEXT: !1 = !{}
@empty = global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !1

; CHECK: sanitize.unpadded.size operand must be an integer constant
; CHECK-NEXT: ptr @not.an.integer
; CHECK-NEXT: !2 = !{!"4"}
@not.an.integer = global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !2

; CHECK: sanitize.unpadded.size must not exceed the size of the global
; CHECK-NEXT: ptr @too.large
; CHECK-NEXT: !3 = !{i64 64}
@too.large = global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !3

!0 = !{i64 4, i64 8}
!1 = !{}
!2 = !{!"4"}
!3 = !{i64 64}
