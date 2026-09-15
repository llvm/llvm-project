; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -target-abi=il32pc64 -mattr=+experimental-y | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -target-abi=l64pc128 -mattr=+experimental-y | FileCheck %s --check-prefixes=CHECK,RV64

;; Previously crashed due to using pointer size rather than index size when computing the initializer of @b

@a = external addrspace(200) global i32
@b = addrspace(200) global iXLen ptrtoaddr (ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @a, i64 8) to iXLen)

; CHECK:     .globl b
; CHECK:     b:
; RV32-NEXT:   .word a+8
; RV32-NEXT:   .size b, 4
; RV64-NEXT:   .quad a+8
; RV64-NEXT:   .size b, 8
