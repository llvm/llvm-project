; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s
; RUN: opt < %s -S -passes='speculative-execution' \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

; Hoist in if-then pattern.
define void @ifThen() {
; CHECK-LABEL: @ifThen(
; CHECK: %x = add i32 2, 3
; CHECK: br i1 true
  br i1 true, label %a, label %b
; CHECK: a:
a:
  %x = add i32 2, 3
; CHECK: br label
  br label %b
; CHECK: b:
b:
; CHECK: ret void
  ret void
}

; Hoist in if-else pattern.
define void @ifElse() {
; CHECK-LABEL: @ifElse(
; CHECK: %x = add i32 2, 3
; CHECK: br i1 true
  br i1 true, label %b, label %a
; CHECK: a:
a:
  %x = add i32 2, 3
; CHECK: br label
  br label %b
; CHECK: b:
b:
; CHECK: ret void
  ret void
}

; Hoist in if-then-else pattern if it is equivalent to if-then.
define void @ifElseThenAsIfThen() {
; CHECK-LABEL: @ifElseThenAsIfThen(
; CHECK: %x = add i32 2, 3
; CHECK: br
  br i1 true, label %a, label %b
; CHECK: a:
a:
  %x = add i32 2, 3
; CHECK: br label
  br label %c
; CHECK: b:
b:
  br label %c
; CHECK: c
c:
  ret void
}

; Hoist in if-then-else pattern if it is equivalent to if-else.
define void @ifElseThenAsIfElse() {
; CHECK-LABEL: @ifElseThenAsIfElse(
; CHECK: %x = add i32 2, 3
; CHECK: br
  br i1 true, label %b, label %a
; CHECK: a:
a:
  %x = add i32 2, 3
; CHECK: br label
  br label %c
; CHECK: b:
b:
  br label %c
; CHECK: c
c:
  ret void
}

; Do not hoist if-then-else pattern if it is not equivalent to if-then
; or if-else.
define void @ifElseThen() {
; CHECK-LABEL: @ifElseThen(
; CHECK: br
  br i1 true, label %a, label %b
; CHECK: a:
a:
; CHECK: %x = add
  %x = add i32 2, 3
; CHECK: br label
  br label %c
; CHECK: b:
b:
; CHECK: %y = add
  %y = add i32 2, 3
  br label %c
; CHECK: c
c:
  ret void
}

; Do not hoist loads and do not hoist an instruction past a definition of
; an operand.
define void @doNotHoistPastDef() {
; CHECK-LABEL: @doNotHoistPastDef(
  br i1 true, label %b, label %a
; CHECK-NOT: load
; CHECK-NOT: add
; CHECK: a:
a:
; CHECK: %def = load
  %def = load i32, i32* null
; CHECK: %use = add
  %use = add i32 %def, 0
  br label %b
; CHECK: b:
b:
  ret void
}

; Case with nothing to speculate.
define void @nothingToSpeculate() {
; CHECK-LABEL: @nothingToSpeculate(
  br i1 true, label %b, label %a
; CHECK: a:
a:
; CHECK: %def = load
  %def = load i32, i32* null
  br label %b
; CHECK: b:
b:
  ret void
}

; Still hoist if an operand is defined before the block or is itself hoisted.
define void @hoistIfNotPastDef() {
; CHECK-LABEL: @hoistIfNotPastDef(
; CHECK: %x = load
  %x = load i32, i32* null
; CHECK: %y = add i32 %x, 1
; CHECK: %z = add i32 %y, 1
; CHECK: br
  br i1 true, label %b, label %a
; CHECK: a:
a:
  %y = add i32 %x, 1
  %z = add i32 %y, 1
  br label %b
; CHECK: b:
b:
  ret void
}

; Do not hoist if the speculation cost is too high.
define void @costTooHigh() {
; CHECK-LABEL: @costTooHigh(
; CHECK: br
  br i1 true, label %b, label %a
; CHECK: a:
a:
; CHECK: %r1 = add
  %r1 = add i32 1, 1
; CHECK: %r2 = add
  %r2 = add i32 1, 1
; CHECK: %r3 = add
  %r3 = add i32 1, 1
; CHECK: %r4 = add
  %r4 = add i32 1, 1
; CHECK: %r5 = add
  %r5 = add i32 1, 1
  br label %b
; CHECK: b:
b:
  ret void
}

; Do not hoist if too many instructions are left behind.
define void @tooMuchLeftBehind() {
; CHECK-LABEL: @tooMuchLeftBehind(
; CHECK: br
  br i1 true, label %b, label %a
; CHECK: a:
a:
; CHECK: %x = load
  %x = load i32, i32* null
; CHECK: %r1 = add
  %r1 = add i32 %x, 1
; CHECK: %r2 = add
  %r2 = add i32 %x, 1
; CHECK: %r3 = add
  %r3 = add i32 %x, 1
  br label %b
; CHECK: b:
b:
  ret void
}
