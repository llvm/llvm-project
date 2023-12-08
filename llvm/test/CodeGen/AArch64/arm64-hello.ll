; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -disable-post-ra -frame-pointer=all | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -disable-post-ra | FileCheck %s --check-prefix=CHECK-LINUX

; CHECK-LABEL: main:
; CHECK:	sub	sp, sp, #32
; CHECK-NEXT:	stp	x29, x30, [sp, #16]
; CHECK-NEXT:	add	x29, sp, #16
; CHECK-NEXT:	stur	wzr, [x29, #-4]
; CHECK:	adrp	x0, l_.str@PAGE
; CHECK:	add	x0, x0, l_.str@PAGEOFF
; CHECK-NEXT:	bl	_puts
; CHECK-NEXT:	ldp	x29, x30, [sp, #16]
; CHECK-NEXT:	add	sp, sp, #32
; CHECK-NEXT:	ret

; CHECK-LINUX-LABEL: main:
; CHECK-LINUX:	str	x30, [sp, #-16]!
; CHECK-LINUX-NEXT:	str	wzr, [sp, #12]
; CHECK-LINUX:	adrp	x0, .L.str
; CHECK-LINUX:	add	x0, x0, :lo12:.L.str
; CHECK-LINUX-NEXT:	bl	puts
; CHECK-LINUX-NEXT:	ldr	x30, [sp], #16
; CHECK-LINUX-NEXT:	ret

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00"

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call i32 @puts(ptr @.str)
  ret i32 %call
}

declare i32 @puts(ptr)
