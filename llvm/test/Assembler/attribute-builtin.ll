
; Make sure that llvm-as/llvm-dis properly assembly/disassembly the 'builtin'
; attribute.
;
; rdar://13727199

; RUN: llvm-as -disable-verify < %s | \
; RUN: llvm-dis | \
; RUN: llvm-as -disable-verify | \
; RUN: llvm-dis | \
; RUN: FileCheck -check-prefix=CHECK-ASSEMBLES %s

; CHECK-ASSEMBLES: declare ptr @foo(ptr) [[NOBUILTIN:#[0-9]+]]
; CHECK-ASSEMBLES: call ptr @foo(ptr %x) [[BUILTIN:#[0-9]+]]
; CHECK-ASSEMBLES: attributes [[NOBUILTIN]] = { nobuiltin }
; CHECK-ASSEMBLES: attributes [[BUILTIN]] = { builtin }

declare ptr @foo(ptr) #1
define ptr @bar(ptr %x) {
  %y = call ptr @foo(ptr %x) #0
  ret ptr %y
}

; Make sure that we do not accept the 'builtin' attribute on function
; definitions, function declarations, and on call sites that call functions
; which do not have nobuiltin on them.
; rdar://13727199

; RUN: not llvm-as <%s 2>&1  | FileCheck -check-prefix=CHECK-BAD %s

; CHECK-BAD: Attribute 'builtin' can only be applied to a callsite.
; CHECK-BAD-NEXT: ptr @car
; CHECK-BAD: Attribute 'builtin' can only be applied to a callsite.
; CHECK-BAD-NEXT: ptr @mar

declare ptr @lar(ptr)

define ptr @har(ptr %x) {
  %y = call ptr @lar(ptr %x) #0
  ret ptr %y
}

define ptr @car(ptr %x) #0 {
  ret ptr %x
}

declare ptr @mar(ptr) #0

attributes #0 = { builtin }
attributes #1 = { nobuiltin }
