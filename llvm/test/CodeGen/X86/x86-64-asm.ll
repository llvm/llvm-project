; RUN: llc < %s
; PR1029

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

define void @frame_dummy() {
entry:
        %tmp1 = tail call ptr (ptr) asm "", "=r,0,~{dirflag},~{fpsr},~{flags}"( ptr null )           ; <ptr> [#uses=0]
        ret void
}

