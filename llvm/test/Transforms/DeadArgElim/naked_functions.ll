; RUN: opt -S -deadargelim %s | FileCheck %s

; Don't eliminate dead arugments from naked functions.
; CHECK: define internal i32 @naked(i32 %x)

define internal i32 @naked(i32 %x) #0 {
  tail call void asm sideeffect inteldialect "mov eax, [esp + $$4]\0A\09ret", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  unreachable
}


; Don't eliminate dead varargs from naked functions.
; CHECK: define internal i32 @naked_va(i32 %x, ...)

define internal i32 @naked_va(i32 %x, ...) #0 {
  tail call void asm sideeffect inteldialect "mov eax, [esp + $$8]\0A\09ret", "~{eax},~{dirflag},~{fpsr},~{flags}"()
  unreachable
}

define i32 @f(i32 %x, i32 %y) {
  %r = call i32 @naked(i32 %x)
  %s = call i32 (i32, ...) @naked_va(i32 %x, i32 %r)

; Make sure the arguments are still there: not removed or replaced with undef.
; CHECK: %r = call i32 @naked(i32 %x)
; CHECK: %s = call i32 (i32, ...) @naked_va(i32 %x, i32 %r)

  ret i32 %s
}

attributes #0 = { naked }
