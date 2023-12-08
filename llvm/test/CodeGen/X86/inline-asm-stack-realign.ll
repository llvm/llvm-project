; RUN: not llc -mtriple=i686-pc-win32 < %s 2>&1 | FileCheck %s

; FIXME: This is miscompiled due to our unconditional use of esi as the base
; pointer.
; XFAIL: *

; CHECK: Stack realignment in presence of dynamic stack adjustments is not supported with inline assembly

define i32 @foo() {
entry:
  %r = alloca i32, align 16
  store i32 -1, ptr %r, align 16
  call void asm sideeffect inteldialect "push esi\0A\09xor esi, esi\0A\09mov dword ptr $0, esi\0A\09pop esi", "=*m,~{flags},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"(ptr %r)
  %0 = load i32, ptr %r, align 16
  ret i32 %0
}
