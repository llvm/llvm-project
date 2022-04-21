; RUN: llc -verify-machineinstrs -enable-machine-outliner -mattr=+m -mtriple=riscv64 < %s | FileCheck %s

; Ensure that we won't outline CFIs when they are needed in unwinding.

define i32 @func1(i32 %x) #0 {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    call t0, OUTLINED_FUNCTION_0
; CHECK-NEXT:    call __cxa_allocate_exception@plt
; CHECK-NEXT:    sw s0, 0(a0)
; CHECK-NEXT:    lui a1, %hi(_ZTIi)
; CHECK-NEXT:    addi a1, a1, %lo(_ZTIi)
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    call __cxa_throw@plt
entry:
  %mul = mul i32 %x, %x
  %add = add i32 %mul, 1
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %0 = bitcast i8* %exception to i32*
  store i32 %add, i32* %0
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable
}

define i32 @func2(i32 %x) #0 {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    call t0, OUTLINED_FUNCTION_0
; CHECK-NEXT:    call __cxa_allocate_exception@plt
; CHECK-NEXT:    sw s0, 0(a0)
; CHECK-NEXT:    lui a1, %hi(_ZTIi)
; CHECK-NEXT:    addi a1, a1, %lo(_ZTIi)
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    call __cxa_throw@plt
entry:
  %mul = mul i32 %x, %x
  %add = add i32 %mul, 1
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %0 = bitcast i8* %exception to i32*
  store i32 %add, i32* %0
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sd ra, 8(sp)
; CHECK-NEXT:    sd s0, 0(sp)
; CHECK-NEXT:    mulw a0, a0, a0
; CHECK-NEXT:    addiw s0, a0, 1
; CHECK-NEXT:    li a0, 4

@_ZTIi = external constant i8*
declare i8* @__cxa_allocate_exception(i64)
declare void @__cxa_throw(i8*, i8*, i8*)

attributes #0 = { minsize noreturn }
