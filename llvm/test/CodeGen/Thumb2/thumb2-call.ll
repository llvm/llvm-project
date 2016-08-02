; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=thumbv7-linux -mattr=+thumb2 | FileCheck %s -check-prefix=LINUX

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
; DARWIN-LABEL: f:
; DARWIN: blx _g

; LINUX-LABEL: f:
; LINUX: bl g
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @h() {
; DARWIN-LABEL: h:
; DARWIN: blx r0

; LINUX-LABEL: h:
; LINUX: blx r0
        %tmp = load i32 ()*, i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}
