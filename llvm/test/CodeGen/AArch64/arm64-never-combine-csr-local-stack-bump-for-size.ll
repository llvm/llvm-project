; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -disable-post-ra | FileCheck %s

; CHECK-LABEL: main:
; CHECK:       stp     x29, x30, [sp, #-16]!
; CHECK-NEXT:  stp     xzr, xzr, [sp, #-16]!
; CHECK:       adrp    x0, l_.str@PAGE
; CHECK:       add     x0, x0, l_.str@PAGEOFF
; CHECK-NEXT:  bl      _puts
; CHECK-NEXT:   add     sp, sp, #16
; CHECK-NEXT:	ldp	x29, x30, [sp], #16
; CHECK-NEXT:	ret

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00"

define i32 @main() nounwind ssp optsize {
entry:
  %local1 = alloca i64, align 8
  %local2 = alloca i64, align 8
  store i64 0, i64* %local1
  store i64 0, i64* %local2
  %call = call i32 @puts(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0))
  ret i32 %call
}

declare i32 @puts(i8*)
