
; Make sure that llvm-as/llvm-dis properly assembly/disassembly the 'builtin'
; attribute.
;
; rdar://13727199

; RUN: llvm-as --opaque-pointers=0 -disable-verify < %s | \
; RUN: llvm-dis --opaque-pointers=0 | \
; RUN: llvm-as --opaque-pointers=0 -disable-verify | \
; RUN: llvm-dis --opaque-pointers=0 | \
; RUN: FileCheck -check-prefix=CHECK-ASSEMBLES %s

; CHECK-ASSEMBLES: declare i8* @foo(i8*) [[NOBUILTIN:#[0-9]+]]
; CHECK-ASSEMBLES: call i8* @foo(i8* %x) [[BUILTIN:#[0-9]+]]
; CHECK-ASSEMBLES: attributes [[NOBUILTIN]] = { nobuiltin }
; CHECK-ASSEMBLES: attributes [[BUILTIN]] = { builtin }

declare i8* @foo(i8*) #1
define i8* @bar(i8* %x) {
  %y = call i8* @foo(i8* %x) #0
  ret i8* %y
}

; Make sure that we do not accept the 'builtin' attribute on function
; definitions, function declarations, and on call sites that call functions
; which do not have nobuiltin on them.
; rdar://13727199

; RUN: not llvm-as <%s 2>&1  | FileCheck -check-prefix=CHECK-BAD %s

; CHECK-BAD: Attribute 'builtin' can only be applied to a callsite.
; CHECK-BAD-NEXT: i8* (i8*)* @car
; CHECK-BAD: Attribute 'builtin' can only be applied to a callsite.
; CHECK-BAD-NEXT: i8* (i8*)* @mar

declare i8* @lar(i8*)

define i8* @har(i8* %x) {
  %y = call i8* @lar(i8* %x) #0
  ret i8* %y
}

define i8* @car(i8* %x) #0 {
  ret i8* %x
}

declare i8* @mar(i8*) #0

attributes #0 = { builtin }
attributes #1 = { nobuiltin }
