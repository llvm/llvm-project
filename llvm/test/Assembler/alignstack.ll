; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

define void @test1() nounwind {
; CHECK: test1
; CHECK: sideeffect
; CHECK-NOT: alignstack
	tail call void asm sideeffect "mov", "~{dirflag},~{fpsr},~{flags}"() nounwind
	ret void
; CHECK: ret
}
define void @test2() nounwind {
; CHECK: test2
; CHECK: sideeffect
; CHECK: alignstack
	tail call void asm sideeffect alignstack "mov", "~{dirflag},~{fpsr},~{flags}"() nounwind
	ret void
; CHECK: ret
}
define void @test3() nounwind {
; CHECK: test3
; CHECK-NOT: sideeffect
; CHECK: alignstack
	tail call void asm alignstack "mov", "~{dirflag},~{fpsr},~{flags}"() nounwind
	ret void
; CHECK: ret
}
define void @test4() nounwind {
; CHECK: test4
; CHECK-NOT: sideeffect
; CHECK-NOT: alignstack
	tail call void asm  "mov", "~{dirflag},~{fpsr},~{flags}"() nounwind
	ret void
; CHECK: ret
}
