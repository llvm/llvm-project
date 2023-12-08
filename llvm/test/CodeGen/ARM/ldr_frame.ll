; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s

; CHECK-LABEL: f1
; CHECK-NOT: mov
define i32 @f1() {
	%buf = alloca [32 x i32], align 4
	%tmp1 = load i32, ptr %buf
	ret i32 %tmp1
}

; CHECK-LABEL: f2
; CHECK-NOT: mov
define i32 @f2() {
	%buf = alloca [32 x i8], align 4
	%tmp1 = load i8, ptr %buf
        %tmp2 = zext i8 %tmp1 to i32
	ret i32 %tmp2
}

; CHECK-LABEL: f3
; CHECK-NOT: mov
define i32 @f3() {
	%buf = alloca [32 x i32], align 4
	%tmp = getelementptr [32 x i32], ptr %buf, i32 0, i32 32
	%tmp1 = load i32, ptr %tmp
	ret i32 %tmp1
}

; CHECK-LABEL: f4
; CHECK-NOT: mov
define i32 @f4() {
	%buf = alloca [32 x i8], align 4
	%tmp = getelementptr [32 x i8], ptr %buf, i32 0, i32 2
	%tmp1 = load i8, ptr %tmp
        %tmp2 = zext i8 %tmp1 to i32
	ret i32 %tmp2
}
