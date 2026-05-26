; RUN: llc -mtriple=xtensa -function-sections --mcpu=esp32 --filetype=obj < %s\
; RUN: | llvm-objdump -r -s --triple=xtensa --mcpu=esp32 -\
; RUN: | FileCheck -check-prefix=CHECK %s

; CHECK-LABEL: RELOCATION RECORDS FOR [.literal.func_b]:
; CHECK:       OFFSET   TYPE                     VALUE
; CHECK-NEXT:  00000000 R_XTENSA_32              func_a

; CHECK-LABEL: Contents of section .literal.func_b:
; CHECK:       0000 00000000                             ....

define i32 @func_a(i32 %x) #0 {
  %1 = alloca i32, align 4
  store i32 %x, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %then, label %else
then:
  %4 = load i32, ptr %1, align 4
  %5 = add i32 %4, 1
  ret i32 %5
else:
  %6 = load i32, ptr %1, align 4
  %7 = sub i32 %6, 1
  ret i32 %7
}

define i32 @func_b() #0 {
  %fp = alloca ptr, align 4
  store ptr @func_a, ptr %fp, align 4
  %1 = load ptr, ptr %fp, align 4
  %2 = call i32 %1(i32 42)
  ret i32 %2
}
