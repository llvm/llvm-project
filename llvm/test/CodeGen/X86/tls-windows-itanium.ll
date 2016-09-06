; RUN: llc -mtriple i686-windows-itanium -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-ASM
; RUN: llc -mtriple i686-windows-itanium -filetype obj -o - %s | llvm-readobj -relocations - | FileCheck %s -check-prefix CHECK-OBJ

@get_count_incremented.count = internal thread_local unnamed_addr global i32 0, align 4

define i32 @get_count_incremented() {
entry:
  %0 = load i32, i32* @get_count_incremented.count, align 4
  %inc = add i32 %0, 1
  store i32 %inc, i32* @get_count_incremented.count, align 4
  ret i32 %inc
}

; CHECK-ASM-LABEL: _get_count_incremented:
; CHECK-ASM: movl __tls_index, %eax
; CHECK-ASM: movl %fs:__tls_array, %ecx
; CHECK-ASM: movl (%ecx,%eax,4), %ecx
; CHECK-ASM: _get_count_incremented.count@SECREL32(%ecx), %eax
; CHECK-ASM: incl %eax
; CHECK-ASM: movl %eax, _get_count_incremented.count@SECREL32(%ecx)
; CHECK-ASM: retl

; CHECK-OBJ: Relocations [
; CHECK-OBJ:   Section ({{[0-9]+}}) .text {
; CHECK-OBJ:     0x1 IMAGE_REL_I386_DIR32 __tls_index
; CHECK-OBJ:     0x8 IMAGE_REL_I386_DIR32 __tls_array
; CHECK-OBJ:     0x11 IMAGE_REL_I386_SECREL _get_count_incremented.count
; CHECK-OBJ:     0x18 IMAGE_REL_I386_SECREL _get_count_incremented.count
; CHECK-OBJ:   }
; CHECK-OBJ: ]
