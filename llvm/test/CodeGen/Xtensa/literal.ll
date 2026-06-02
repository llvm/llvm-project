; RUN: llc -mtriple=xtensa -function-sections --mcpu=esp32 --filetype=obj < %s\
; RUN: | llvm-objdump -r -s --triple=xtensa --mcpu=esp32 -\
; RUN: | FileCheck %s

; CHECK-LABEL: RELOCATION RECORDS FOR [.literal.func_b]:
; CHECK:       OFFSET   TYPE                     VALUE
; CHECK-NEXT:  00000000 R_XTENSA_32              func_a

; CHECK-LABEL: Contents of section .literal.func_b:
; CHECK:       0000 00000000                             ....

define i32 @func_b() #0 {
  %fp = alloca ptr, align 4
  store ptr @func_a, ptr %fp, align 4
  %func_a_ptr = load ptr, ptr %fp, align 4
  %res = call i32 %func_a_ptr(i32 42)
  ret i32 %res
}

declare dso_local i32 @func_a(i32 %x)
