; RUN: llc -filetype=asm %s -o - | FileCheck %s

; CHECK:      movw r0, :lower16:a
; CHECK-NEXT: movt r0, :upper16:a
; CHECK-NEXT: vldr s6, [r0]

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux-gnueabihf"

@a = local_unnamed_addr global i32 0, align 4

define void @_Z1dv() local_unnamed_addr {
entry:
  %0 = load i32, ptr @a, align 4
  tail call void asm sideeffect "", "{s6}"(i32 %0)
  ret void
}
