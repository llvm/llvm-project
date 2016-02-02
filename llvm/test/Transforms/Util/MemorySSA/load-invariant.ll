; XFAIL:
; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Invariant loads should be considered live on entry, because, once the
; location is known to be dereferenceable, the value can never change.
;
; Currently XFAILed because this optimization was held back from the initial
; commit.

@g = external global i32

declare void @clobberAllTheThings()

define i32 @foo() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* @g, align 4, !invariant.load !0
  ret i32 %1
}

!0 = !{}
