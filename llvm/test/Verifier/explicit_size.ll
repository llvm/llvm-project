; RUN: split-file %s %t
; RUN: not opt -verify -disable-output < %t/multiple.ll 2>&1 | FileCheck %s --check-prefix=MULTIPLE
; RUN: not opt -verify -disable-output < %t/operands.ll 2>&1 | FileCheck %s --check-prefix=OPERANDS
; RUN: not opt -verify -disable-output < %t/invalid_md.ll 2>&1 | FileCheck %s --check-prefix=INVALID_MD
; RUN: not opt -verify -disable-output < %t/not_int.ll 2>&1 | FileCheck %s --check-prefix=NOT_INT

; MULTIPLE: only one !explicit_size can be attached to a global variable
;--- multiple.ll
@a = global { i32, [28 x i8] } zeroinitializer, align 32, !explicit_size !0, !explicit_size !0

!0 = !{i64 4}

; OPERANDS: !explicit_size must have exactly one operand
;--- operands.ll
@a = global { i32, [28 x i8] } zeroinitializer, align 32, !explicit_size !0

!0 = !{i64 4, i64 4}

; INVALID_MD: !explicit_size operand must be a value
;--- invalid_md.ll
@a = global { i32, [28 x i8] } zeroinitializer, align 32, !explicit_size !0

!0 = !{!1}
!1 = !{i64 4}

; NOT_INT: !explicit_size value must be an integer
;--- not_int.ll
@a = global { i32, [28 x i8] } zeroinitializer, align 32, !explicit_size !0
declare i32 @b()

!0 = !{i32 ()* @b}
