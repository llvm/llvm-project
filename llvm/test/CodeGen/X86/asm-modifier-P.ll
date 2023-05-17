; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -relocation-model=pic    | FileCheck %s -check-prefix=CHECK-PIC-32
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -relocation-model=static | FileCheck %s -check-prefix=CHECK-STATIC-32
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s -check-prefix=CHECK-STATIC-64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic    | FileCheck %s -check-prefix=CHECK-PIC-64
; PR3379
; XFAIL: *

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
@G = external global i32              ; <ptr> [#uses=1]

declare void @bar(...)

; extern int G;
; void test1() {
;  asm("frob %0 x" : : "m"(G));
;  asm("frob %P0 x" : : "m"(G));
;}

define void @test1() nounwind {
entry:
; P suffix removes (rip) in -static 64-bit mode.

; CHECK-PIC-64-LABEL: test1:
; CHECK-PIC-64: movq	G@GOTPCREL(%rip), %rax
; CHECK-PIC-64: frob (%rax) x
; CHECK-PIC-64: frob (%rax) x

; CHECK-STATIC-64-LABEL: test1:
; CHECK-STATIC-64: frob G(%rip) x
; CHECK-STATIC-64: frob G x

; CHECK-PIC-32-LABEL: test1:
; CHECK-PIC-32: frob G x
; CHECK-PIC-32: frob G x

; CHECK-STATIC-32-LABEL: test1:
; CHECK-STATIC-32: frob G x
; CHECK-STATIC-32: frob G x

        call void asm "frob $0 x", "*m"(ptr @G) nounwind
        call void asm "frob ${0:P} x", "*m"(ptr @G) nounwind
        ret void
}

define void @test3() nounwind {
entry:
; CHECK-STATIC-64-LABEL: test3:
; CHECK-STATIC-64: call bar
; CHECK-STATIC-64: call test3
; CHECK-STATIC-64: call $bar
; CHECK-STATIC-64: call $test3

; CHECK-STATIC-32-LABEL: test3:
; CHECK-STATIC-32: call bar
; CHECK-STATIC-32: call test3
; CHECK-STATIC-32: call $bar
; CHECK-STATIC-32: call $test3

; CHECK-PIC-64-LABEL: test3:
; CHECK-PIC-64: call bar@PLT
; CHECK-PIC-64: call test3@PLT
; CHECK-PIC-64: call $bar
; CHECK-PIC-64: call $test3

; CHECK-PIC-32-LABEL: test3:
; CHECK-PIC-32: call bar@PLT
; CHECK-PIC-32: call test3@PLT
; CHECK-PIC-32: call $bar
; CHECK-PIC-32: call $test3


; asm(" blah %P0" : : "X"(bar));
  tail call void asm sideeffect "call ${0:P}", "X"(ptr @bar) nounwind
  tail call void asm sideeffect "call ${0:P}", "X"(ptr @test3) nounwind
  tail call void asm sideeffect "call $0", "X"(ptr @bar) nounwind
  tail call void asm sideeffect "call $0", "X"(ptr @test3) nounwind
  ret void
}
