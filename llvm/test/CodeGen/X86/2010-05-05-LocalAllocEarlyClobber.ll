; RUN: llc < %s -O0 -regalloc=fast -no-integrated-as | FileCheck %s
; PR6520

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

%0 = type { ptr, ptr, i32 }

define ptr @func() nounwind ssp {
entry:
  %retval = alloca ptr, align 4                   ; <ptr> [#uses=2]
  %ret = alloca ptr, align 4                      ; <ptr> [#uses=2]
  %p = alloca ptr, align 4                        ; <ptr> [#uses=1]
  %t = alloca i32, align 4                        ; <ptr> [#uses=1]
; The earlyclobber $1 should only appear once. It should not be shared.
; CHECK: deafbeef, [[REG:%e.x]]
; CHECK-NOT: [[REG]]
; CHECK: InlineAsm End
  %0 = call %0 asm "mov    $$0xdeafbeef, $1\0A\09mov    $$0xcafebabe, $0\0A\09mov    $0, $2\0A\09", "=&r,=&r,=&{cx},~{dirflag},~{fpsr},~{flags}"() nounwind, !srcloc !0 ; <%0> [#uses=3]
  %asmresult = extractvalue %0 %0, 0              ; <ptr> [#uses=1]
  %asmresult1 = extractvalue %0 %0, 1             ; <ptr> [#uses=1]
  %asmresult2 = extractvalue %0 %0, 2             ; <i32> [#uses=1]
  store ptr %asmresult, ptr %ret
  store ptr %asmresult1, ptr %p
  store i32 %asmresult2, ptr %t
  %tmp = load ptr, ptr %ret                           ; <ptr> [#uses=1]
  store ptr %tmp, ptr %retval
  %1 = load ptr, ptr %retval                          ; <ptr> [#uses=1]
  ret ptr %1
}

!0 = !{i32 79}
