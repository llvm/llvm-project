; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; RUN: llc -mtriple=armv6-linux-gnueabi %s -o - | FileCheck %s

; RUN: llc -mtriple=armv6-linux-gnu --frame-pointer=all %s -o - \
; RUN:  | FileCheck %s -check-prefix CHECK-FP-ELIM

; RUN: llc -mtriple=armv6-apple-ios %s -o - \
; RUN:  | FileCheck %s -check-prefix CHECK-FP-ELIM

@str = internal constant [12 x i8] c"Hello World\00"

define i32 @main() "frame-pointer"="all" {
	%tmp = call i32 @puts( i8* getelementptr ([12 x i8], [12 x i8]* @str, i32 0, i64 0) )		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @puts(i8*)

; CHECK-LABEL: main
; CHECK-NOT: mov
; CHECK: mov r11, sp
; CHECK-NOT: mov
; CHECK: mov r0, #0
; CHECK-NOT: mov

; CHECK-FP-ELIM-LABEL: main
; CHECK-FP-ELIM: mov
; CHECK-FP-ELIM: mov
; CHECK-FP-ELIM-NOT: mov

