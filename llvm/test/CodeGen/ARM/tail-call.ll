; RUN: llc -mtriple armv7 -target-abi apcs -O0 -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-TAIL -check-prefix CHECK
; RUN: llc -mtriple armv7 -target-abi apcs -O0 -disable-tail-calls -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-NO-TAIL -check-prefix CHECK
; RUN: llc -mtriple armv7 -target-abi aapcs -O0 -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-TAIL-AAPCS -check-prefix CHECK

declare i32 @callee(i32 %i)
declare extern_weak fastcc void @callee_weak()

define i32 @caller(i32 %i) {
entry:
  %r = tail call i32 @callee(i32 %i)
  ret i32 %r
}

; CHECK-TAIL-LABEL: caller
; CHECK-TAIL: b callee

; CHECK-NO-TAIL-LABEL: caller
; CHECK-NO-TAIL: push {lr}
; CHECK-NO-TAIL: bl callee
; CHECK-NO-TAIL: pop {lr}
; CHECK-NO-TAIL: bx lr


; Weakly-referenced extern functions cannot be tail-called, as AAELF does
; not define the behaviour of branch instructions to undefined weak symbols.
define fastcc void @caller_weak() {
; CHECK-LABEL: caller_weak:
; CHECK: bl callee_weak
  tail call void @callee_weak()
  ret void
}

; A tail call can be optimized if all the arguments can be passed in registers
; R0-R3, or the remaining arguments are already in the caller's parameter area
; in the stack. Variadic functions are no different.
declare i32 @variadic(i32, ...)

; e.g. four integers
define void @v_caller_ints1(i32 %a, i32 %b) {
; CHECK-LABEL: v_caller_ints1:
; CHECK-TAIL: b variadic
; CHECK-TAIL-AAPCS: b variadic
; CHECK-NO-TAIL: bl variadic
entry:
  %call = tail call i32 (i32, ...) @variadic(i32 %a, i32 %b, i32 %b, i32 %a)
  ret void
}

; e.g. two 32-bit integers, one 64-bit integer (needs to span two regs)
define void @v_caller_ints2(i32 %y, i64 %z) {
; CHECK-LABEL: v_caller_ints2:
; CHECK-TAIL: b variadic
; CHECK-TAIL-AAPCS: b variadic
; CHECK-NO-TAIL: bl variadic
entry:
  %call = tail call i32 (i32, ...) @variadic(i32 %y, i32 %y, i64 %z)
  ret void
}

; e.g. two 32-bit integers, one 64-bit integer (needs to span two regs). Notice
; that %z is passed in r1-r2 if APCS is used, contrary to AAPCS where r2-r3
; would be used (since double-word types must start at an even register). In the
; latter case, the third argument needs to be passed through the stack.
define void @v_caller_ints3(i32 %y, i64 %z) {
; CHECK-LABEL: v_caller_ints3:
; CHECK-TAIL: b variadic
; CHECK-TAIL-AAPCS: bl variadic
; CHECK-NO-TAIL: bl variadic
entry:
  %call = tail call i32 (i32, ...) @variadic(i32 %y, i64 %z, i32 %y)
  ret void
}

; e.g. two 32-bit integers, one 64-bit integer and another 64-bit integer that
; doesn't fit in r0-r3 but comes from the caller argument list and is in the
; same position.
define void @v_caller_ints4(i64 %a, i32 %b, i32 %c, i64 %d) {
; CHECK-LABEL: v_caller_ints4:
; CHECK-TAIL: b variadic
; CHECK-TAIL-AAPCS: b variadic
; CHECK-NO-TAIL: bl variadic
entry:
  %call = tail call i32 (i32, ...) @variadic(i32 %b, i32 %c, i64 %a, i64 %d)
  ret void
}

; If the arguments do not fit in r0-r3 and the existing parameters cannot be
; taken from the caller's parameter region, the optimization is not supported.

; e.g. one 32-bit integer, two 64-bit integers
define void @v_caller_ints_fail(i32 %y, i64 %z) {
; CHECK-LABEL: v_caller_ints_fail:
; CHECK: bl variadic
entry:
  %call = tail call i32 (i32, ...) @variadic(i32 %y, i64 %z, i64 %z)
  ret void
}

; Check that nonnull attributes don't inhibit tailcalls.

declare nonnull i8* @nonnull_callee(i8* %p, i32 %val)
define i8* @nonnull_caller(i8* %p, i32 %val) {
; CHECK-LABEL: nonnull_caller:
; CHECK-TAIL: b nonnull_callee
; CHECK-NO-TAIL: bl nonnull_callee
entry:
  %call = tail call i8* @nonnull_callee(i8* %p, i32 %val)
  ret i8* %call
}

; Check that noalias attributes don't inhibit tailcalls.

declare noalias i8* @noalias_callee(i8* %p, i32 %val)
define i8* @noalias_caller(i8* %p, i32 %val) {
; CHECK-LABEL: noalias_caller:
; CHECK-TAIL: b noalias_callee
; CHECK-NO-TAIL: bl noalias_callee
entry:
  %call = tail call i8* @noalias_callee(i8* %p, i32 %val)
  ret i8* %call
}


; Check that alignment attributes don't inhibit tailcalls.

declare align 8 i8* @align8_callee(i8* %p, i32 %val)
define i8* @align8_caller(i8* %p, i32 %val) {
; CHECK-LABEL: align8_caller:
; CHECK-TAIL: b align8_callee
; CHECK-NO-TAIL: bl align8_callee
entry:
  %call = tail call i8* @align8_callee(i8* %p, i32 %val)
  ret i8* %call
}
