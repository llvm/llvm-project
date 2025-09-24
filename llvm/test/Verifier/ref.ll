; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

@a = global i32 1, !ref !0
@b = global i32 2, !ref !1
@c = global i32 3, !ref !1, !ref !2
@d = global i32 4, !ref !3

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

