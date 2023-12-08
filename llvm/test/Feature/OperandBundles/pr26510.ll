; RUN: opt -S -passes='require<globals-aa>,function-attrs' < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s

; Apart from checking for the direct cause of the bug, we also check
; if any problematic aliasing rules have accidentally snuck into -O3.
;
; Since the "abc" operand bundle is not a special operand bundle that
; LLVM knows about, all of the stores and loads in @test below have to
; stay.

declare void @foo() readnone

; CHECK-LABEL: define ptr @test(ptr %p)
; CHECK:   %a = alloca ptr, align 8
; CHECK:   store ptr %p, ptr %a, align 8
; CHECK:   call void @foo() [ "abc"(ptr %a) ]
; CHECK:   %reload = load ptr, ptr %a, align 8
; CHECK:   ret ptr %reload
; CHECK: }

define ptr @test(ptr %p) {
  %a = alloca ptr, align 8
  store ptr %p, ptr %a, align 8
  call void @foo() ["abc" (ptr %a)]
  %reload = load ptr, ptr %a, align 8
  ret ptr %reload
}
