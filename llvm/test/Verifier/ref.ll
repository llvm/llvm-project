; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 1, !implicit.ref !0
@b = global i32 2, !implicit.ref !1
@c = global i32 3, !implicit.ref !1, !implicit.ref !2
@d = global i32 4, !implicit.ref !3
@e = external global i32, !implicit.ref !1

!0 = !{i32 1}
!1 = !{ptr @b}
!2 = !{!"Hello World!"}
!3 = !{ptr @c, ptr @a}

; CHECK: ref value must be pointer typed
; CHECK: ptr @a
; CHECK: !0 = !{i32 1}

; CHECK: values should not reference themselves
; CHECK: ptr @b
; CHECK: !1 = !{ptr @b}

; CHECK: ref metadata must be ValueAsMetadata
; CHECK: ptr @c
; CHECK: !2 = !{!"Hello World!"}

; CHECK: ref metadata must have one operand
; CHECK: ptr @d
; CHECK: !3 = !{ptr @c, ptr @a}

; CHECK: ref metadata must not be placed on a declaration
; CHECK: @e
