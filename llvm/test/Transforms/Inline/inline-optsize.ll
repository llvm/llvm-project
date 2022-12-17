; RUN: opt -S -Oz < %s | FileCheck %s -check-prefix=OZ
; RUN: opt -S -O2 < %s | FileCheck %s -check-prefix=O2
; RUN: opt -S -Os < %s | FileCheck %s -check-prefix=OS

; The inline threshold for a function with the optsize attribute is currently
; the same as the global inline threshold for -Os. Check that the optsize
; function attribute doesn't alter the function-specific inline threshold if the
; global inline threshold is lower (as for -Oz).

@a = global i32 4

; This function should be larger than the inline threshold for -Oz (25), but
; smaller than the inline threshold for optsize (75).
define i32 @inner() {
  call void @extern()
  %a1 = load volatile i32, ptr @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, ptr @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, ptr @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, ptr @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, ptr @a
  %x5 = add i32 %x3, %a5
  ret i32 %x5
}

; @inner() should be inlined for -O2 and -Os but not for -Oz.
; OZ: call
; O2-NOT: call
; OS-NOT: call
define i32 @outer() optsize {
   %r = call i32 @inner()
   ret i32 %r
}

; @inner() should not be inlined for -O2, -Os and -Oz.
; OZ: call
; O2: call
; OS: call
define i32 @outer2() minsize {
   %r = call i32 @inner()
   ret i32 %r
}

declare void @extern()