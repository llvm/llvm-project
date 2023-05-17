; RUN: llc -mtriple=x86_64-linux -fast-isel -show-mc-encoding < %s | FileCheck %s

; pr22854
; CHECK: addl	$42, %esi               # encoding: [0x83,0xc6,0x2a]

define void @foo(ptr%s, i32 %x) {
  %y = add nsw i32 %x, 42
  store i32 %y, ptr %s, align 4
  ret void
}
