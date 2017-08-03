; RUN: llc < %s -march=x86 -no-integrated-as | FileCheck %s

; ModuleID = 'shant.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define void @f() nounwind {
; CHECK-LABEL: f:
; CHECK-NOT: ret
; CHECK: foo $-81920
; CHECK-NOT: ret
; CHECK: foo $-81920
; CHECK-NOT: ret
; CHECK: foo $-81920
; CHECK-NOT: ret
; CHECK: foo $4294885376
; CHECK: ret

	call void asm sideeffect "foo $0", "n,~{dirflag},~{fpsr},~{flags}"(i32 -81920) nounwind
	call void asm sideeffect "foo $0", "i,~{dirflag},~{fpsr},~{flags}"(i32 -81920) nounwind
	call void asm sideeffect "foo $0", "e,~{dirflag},~{fpsr},~{flags}"(i32 -81920) nounwind
	call void asm sideeffect "foo $0", "Z,~{dirflag},~{fpsr},~{flags}"(i64 4294885376) nounwind
	ret void
}
