; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s

; Without any typed operations, always use the smaller xorps.
; CHECK: test
; CHECK: xorps
define <2 x double> @test() {
	ret <2 x double> zeroinitializer
}

; Prefer a constant pool load here.
; CHECK: test2
; CHECK-NOT: shuf
; CHECK: movaps {{.*}}CPI
define <4 x i32> @test2() nounwind  {
	ret <4 x i32> < i32 0, i32 0, i32 1, i32 0 >
}
