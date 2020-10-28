; RUN: llc  %s -mtriple=armv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -S --symbols - | FileCheck  -check-prefix=OBJ %s
; RUN: llc  %s -mtriple=armv7-linux-gnueabi -o - | \
; RUN:    FileCheck  -check-prefix=ASM %s


@dummy = internal global i32 666
@array00 = internal global [80 x i8] zeroinitializer, align 1
@sum = internal global i32 55
@STRIDE = internal global i32 8

; ASM:          .type   array00,%object         @ @array00
; ASM-NEXT:     .local  array00
; ASM-NEXT:     .comm   array00,80,1
; ASM-NEXT:     .type   sum,%object  @ @sum


; OBJ:      Symbols [
; OBJ:        Symbol {
; OBJ:          Name: array00
; OBJ-NEXT:     Value: 0x0
; OBJ-NEXT:     Size: 80
; OBJ-NEXT:     Binding: Local
; OBJ-NEXT:     Type: Object
; OBJ-NEXT:     Other: 0
; OBJ-NEXT:     Section: .bss

define i32 @main(i32 %argc) nounwind {
  %1 = load i32, i32* @sum, align 4
  %2 = getelementptr [80 x i8], [80 x i8]* @array00, i32 0, i32 %argc
  %3 = load i8, i8* %2
  %4 = zext i8 %3 to i32
  %5 = add i32 %1, %4
  ret i32 %5
}
