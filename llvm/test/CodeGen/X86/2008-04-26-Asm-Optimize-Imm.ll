; RUN: llc -no-integrated-as < %s | FileCheck %s
; rdar://5720231
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @test() nounwind  {
; CHECK-LABEL: test:
; CHECK-NOT: ret
; CHECK: 1 $2 3
; CHECK: ret

	tail call void asm sideeffect " ${0:c} $1 ${2:c} ", "imr,imr,i,~{dirflag},~{fpsr},~{flags}"( i32 1, i32 2, i32 3 ) nounwind 
	ret void
}

