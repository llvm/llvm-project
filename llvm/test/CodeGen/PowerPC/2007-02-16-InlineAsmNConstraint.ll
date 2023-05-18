; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu"

; KB: FIXME: Need to figure out what this should be checking for (or whether test should be removed)
; CHECK: blargh
define void @blargh() {
entry:
	%tmp4 = call i32 asm "rlwimi $0,$2,$3,$4,$5", "=r,0,r,n,n,n"( i32 0, i32 0, i32 0, i32 24, i32 31 )		; <i32> [#uses=0]
	unreachable
}
