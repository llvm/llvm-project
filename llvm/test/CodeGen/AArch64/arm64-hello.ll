; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -disable-post-ra | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -disable-post-ra | FileCheck %s --check-prefix=CHECK-LINUX

; CHECK-LABEL: main:
; CHECK:	stp	x29, x30, [sp, #-16]!
; CHECK-NEXT:	mov	x29, sp
; CHECK-NEXT:	sub	sp, sp, #16
; CHECK-NEXT:	stur	wzr, [x29, #-4]
; CHECK:	adrp	x0, L_.str@PAGE
; CHECK:	add	x0, x0, L_.str@PAGEOFF
; CHECK-NEXT:	bl	_puts
; CHECK-NEXT:	mov	sp, x29
; CHECK-NEXT:	ldp	x29, x30, [sp], #16
; CHECK-NEXT:	ret

; CHECK-LINUX-LABEL: main:
; CHECK-LINUX:	stp	x29, x30, [sp, #-16]!
; CHECK-LINUX-NEXT:	mov	x29, sp
; CHECK-LINUX-NEXT:	sub	sp, sp, #16
; CHECK-LINUX-NEXT:	stur	wzr, [x29, #-4]
; CHECK-LINUX:	adrp	x0, .L.str
; CHECK-LINUX:	add	x0, x0, :lo12:.L.str
; CHECK-LINUX-NEXT:	bl	puts
; CHECK-LINUX-NEXT:	mov	sp, x29
; CHECK-LINUX-NEXT:	ldp	x29, x30, [sp], #16
; CHECK-LINUX-NEXT:	ret

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00"

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @puts(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0))
  ret i32 %call
}

declare i32 @puts(i8*)
