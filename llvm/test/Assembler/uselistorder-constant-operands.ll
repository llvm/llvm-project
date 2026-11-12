; RUN: opt -S -preserve-ll-uselistorder < %s | FileCheck %s
; RUN: verify-uselistorder %s

; A constant that uses a global more than once gets these uses in operand
; order, also when the global is defined after the constant and the parser
; replaces a forward reference. Only @swapped needs a directive.

; CHECK-NOT: {{^}}uselistorder
; CHECK: {{^}}uselistorder ptr @swapped, { 1, 0 }
; CHECK-NOT: {{^}}uselistorder

@self = global { ptr, ptr } { ptr @self, ptr @self }
@fwd.user = global { ptr, ptr, ptr } { ptr @fwd, ptr getelementptr (i8, ptr @fwd, i64 1), ptr @fwd }
@fwd = global i8 0
@fn.user = global [2 x ptr] [ptr @fn, ptr @fn]
@swapped = global { ptr, ptr } { ptr @swapped, ptr @swapped }

define void @fn() {
  ret void
}

uselistorder ptr @swapped, { 1, 0 }
