; Test that a missing global prefix (in this cas a "printf" with no leading '_'
; on Darwin) yields a symbols-not-found error, rather than silently binding
; 'rintf' from math.h.
;
; RUN: not lli -jit-kind=orc %s 2>&1 | FileCheck %s
;
; REQUIRES: system-darwin
; CHECK: Symbols not found: [ printf ]

; ModuleID = 'hw.c'
@.str = private constant [15 x i8] c"hello, world!\0A\00"

define i32 @main(i32 noundef %argc, ptr noundef %argv) {
entry:
  %call = call i32 (ptr, ...) @"\01printf"(ptr noundef @.str)
  ret i32 42
}

declare i32 @"\01printf"(ptr noundef, ...)
