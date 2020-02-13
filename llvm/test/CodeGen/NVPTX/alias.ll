; RUN: not --crash llc < %s -march=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that llc dies gracefully when given an alias.

define i32 @a() { ret i32 0 }
; CHECK: ERROR: Module has aliases
@b = internal alias i32 (), i32 ()* @a
