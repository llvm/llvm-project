; RUN: llc < %s -mtriple=xcore | FileCheck %s
declare i32 @llvm.xcore.getid()

define i32 @test() {
; CHECK-LABEL: test:
; CHECK: get r11, id
; CHECK-NEXT: mov r0, r11
	%result = call i32 @llvm.xcore.getid()
	ret i32 %result
}
