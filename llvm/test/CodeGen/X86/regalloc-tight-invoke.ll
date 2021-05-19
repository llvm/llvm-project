; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s
declare void @foo(i32, ...)

declare i32 @__gxx_personality_v0(...)

; We were running out of registers for this invoke, because:

;     1. The lshr/and pattern gets matched to a no-REX MOV so that ah/bh/... can
;        be used instead, cutting available registers for %b.arg down to eax, ebx,
;        ecx, edx, esi, edi.
;     2. We have a base pointer taking ebx out of contention.
;     3. The landingpad block convinced us we should be defining rax here.
;     3. The al fiddling for the varargs call only noted down that al was spillable,
;        not ah or hax.
;
; So by the time we need to allocate a register for the call all registers are
; tied up and unspillable.

; CHECK-LABEL: bar:
; CHECK: xorl %edi, %edi
; CHECK: movb %dil, {{[0-9]+}}(%rbx)
; CHECK: movb {{[0-9]+}}(%rbx), %al

define i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d, ...) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %mem = alloca i32, i32 %a, align 32   ; Force rbx to be used as a base pointer
  %b.tmp = lshr i32 %b, 8
  %b.arg = and i32 %b.tmp, 255
  invoke void(i32, ...) @foo(i32 42, i32* %mem, i32 %c, i32 %d, i32 %b.arg) to label %success unwind label %fail

success:
  ret i32 0
fail:
  %exc = landingpad { i8*, i32 } cleanup
  %res = extractvalue { i8*, i32 } %exc, 1
  ret i32 %res
}

; CHECK-LABEL: live:
; CHECK: movl {{%.*}}, %eax

define i32 @live(i32 %a, i32 %b, i32 %c, i32 %d, ...) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %mem = alloca i32, i32 %a, align 32   ; Force rbx to be used as a base pointer
  %b.tmp = lshr i32 %b, 8
  %b.arg = and i32 %b.tmp, 255
  invoke void(i32, ...) @foo(i32 42) to label %success unwind label %fail

success:
  ret i32 0
fail:
  %exc = landingpad { i8*, i32 } cleanup
  %res = extractvalue { i8*, i32 } %exc, 1
  ret i32 %b.arg
}
