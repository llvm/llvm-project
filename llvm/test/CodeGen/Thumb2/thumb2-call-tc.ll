; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=thumbv7-linux -mattr=+thumb2 | FileCheck %s -check-prefix=LINUX
; XFAIL: *

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; DARWIN: f:
; DARWIN: blx _g

; LINUX: f:
; LINUX: bl g
        tail call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @h() {
; DARWIN: h:
; DARWIN: bx r0 @ TAILCALL

; LINUX: h:
; LINUX: bx r0 @ TAILCALL
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}

define void @j() {
; DARWIN: j:
; DARWIN: b.w _f  @ TAILCALL

; LINUX: j:
; LINUX: b.w f  @ TAILCALL
        tail call void @f()
        ret void
}
