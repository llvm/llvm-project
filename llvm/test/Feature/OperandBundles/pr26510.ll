; RUN: opt -S -globals-aa -function-attrs < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s

; Apart from checking for the direct cause of the bug, we also check
; if any problematic aliasing rules have accidentally snuck into -O3.
;
; Since the "abc" operand bundle is not a special operand bundle that
; LLVM knows about, all of the stores and loads in @test below have to
; stay.

declare void @foo() readnone

; CHECK-LABEL: define i8* @test(i8* %p)
; CHECK:   %a = alloca i8*, align 8
; CHECK:   store i8* %p, i8** %a, align 8
; CHECK:   call void @foo() [ "abc"(i8** %a) ]
; CHECK:   %reload = load i8*, i8** %a, align 8
; CHECK:   ret i8* %reload
; CHECK: }

define i8* @test(i8* %p) {
  %a = alloca i8*, align 8
  store i8* %p, i8** %a, align 8
  call void @foo() ["abc" (i8** %a)]
  %reload = load i8*, i8** %a, align 8
  ret i8* %reload
}
