; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @test(i32* %X, i32 %B) {
; CHECK: test:
; CHECK-NOT: ret
; CHECK-NOT: lea
; CHECK: mov{{.}} $4, ({{.*}},{{.*}},4)
; CHECK: ret
; CHECK: mov{{.}} ({{.*}},{{.*}},4),
; CHECK: ret

	; This gep should be sunk out of this block into the load/store users.
	%P = getelementptr i32* %X, i32 %B
	%G = icmp ult i32 %B, 1234
	br i1 %G, label %T, label %F
T:
	store i32 4, i32* %P
	ret i32 141
F:
	%V = load i32* %P
	ret i32 %V
}
