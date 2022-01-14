; RUN: llc -verify-machineinstrs -mtriple=armebv8-linux-gnueabi -O0 %s -o - | FileCheck %s

@x = global i64 10, align 8
@y = global i64 20, align 8
@z = global i64 20, align 8

; CHECK_LABEL:	main:
; CHECK:	ldr [[R2:r[0-9]+]], [[[R1:r[0-9]+]]]
; CHECK-NEXT:	ldr [[R1]], [[[R1]], #4]
; CHECK:	mov [[R4:r[0-9]+]], [[R1]]
; CHECK:	ldr [[R5:r[0-9]+]], [[[R1]]]
; CHECK-NEXT:	ldr [[R6:r[0-9]+]], [[[R1]], #4]
; CHECK:	mov [[R7:r[0-9]+]], [[R6]]

define arm_aapcs_vfpcc i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load i64, i64* @z, align 8
  %1 = load i64, i64* @x, align 8
  %2 = cmpxchg i64* @y, i64 %0, i64 %1 seq_cst seq_cst
  %3 = extractvalue { i64, i1 } %2, 1
  ret i32 0
}
