; RUN: not llubi --entry-function=integer < %s 2>&1 | FileCheck %s --check-prefix=INTEGER
; RUN: not llubi --entry-function=floating < %s 2>&1 | FileCheck %s --check-prefix=FLOAT
; RUN: not llubi --entry-function=count < %s 2>&1 | FileCheck %s --check-prefix=COUNT
; RUN: not llubi --entry-function=large_integer_type < %s 2>&1 | FileCheck %s --check-prefix=LARGE_INTEGER_TYPE
; RUN: not llubi --entry-function=small_integer_type < %s 2>&1 | FileCheck %s --check-prefix=SMALL_INTEGER_TYPE
; RUN: not llubi --entry-function=wide_pointer < %s 2>&1 | FileCheck %s --check-prefix=WIDE_POINTER
; RUN: not llubi --entry-function=wide_string < %s 2>&1 | FileCheck %s --check-prefix=WIDE_STRING

target datalayout = "e-p:128:128"

@fmt_integer = constant [3 x i8] c"%d\00"
@fmt_float = constant [3 x i8] c"%f\00"
@fmt_count = constant [3 x i8] c"%n\00"
@fmt_pointer = constant [3 x i8] c"%p\00"
@fmt_string = constant [3 x i8] c"%s\00"

declare i32 @printf(ptr, ...)

define void @integer() {
  call i32 (ptr, ...) @printf(ptr @fmt_integer, ptr null)
  ret void
}

; INTEGER: Immediate UB detected: Argument type mismatch in printf for format specifier 'd' at argument index 1.
; INTEGER-NEXT: error: Execution of function 'integer' failed.

define void @floating() {
  call i32 (ptr, ...) @printf(ptr @fmt_float, i32 0)
  ret void
}

; FLOAT: Immediate UB detected: Argument type mismatch in printf for format specifier 'f' at argument index 1.
; FLOAT-NEXT: error: Execution of function 'floating' failed.

define void @count() {
  call i32 (ptr, ...) @printf(ptr @fmt_count, i32 0)
  ret void
}

; COUNT: Immediate UB detected: Argument type mismatch in printf for format specifier 'n' at argument index 1.
; COUNT-NEXT: error: Execution of function 'count' failed.

define void @large_integer_type() {
  %r = call i32 (ptr, ...) @printf(ptr @fmt_integer, i128 18446744073709551616)
  ret void
}

; LARGE_INTEGER_TYPE: Immediate UB detected: Argument type mismatch in printf for format specifier 'd' at argument index 1.
; LARGE_INTEGER_TYPE-NEXT: error: Execution of function 'large_integer_type' failed.

define void @small_integer_type() {
  %r = call i32 (ptr, ...) @printf(ptr @fmt_integer, i8 8)
  ret void
}

; SMALL_INTEGER_TYPE: Immediate UB detected: Argument type mismatch in printf for format specifier 'd' at argument index 1.
; SMALL_INTEGER_TYPE-NEXT: error: Execution of function 'small_integer_type' failed.

define void @wide_pointer() {
  %p = inttoptr i128 1267650600228229401496703205376 to ptr
  call i32 (ptr, ...) @printf(ptr @fmt_pointer, ptr %p)
  ret void
}

; WIDE_POINTER: Immediate UB detected: Argument type mismatch in printf for format specifier 'p' at argument index 1.
; WIDE_POINTER-NEXT: error: Execution of function 'wide_pointer' failed.

define void @wide_string() {
  %p = inttoptr i128 1267650600228229401496703205376 to ptr
  call i32 (ptr, ...) @printf(ptr @fmt_string, ptr %p)
  ret void
}

; WIDE_STRING: Immediate UB detected: Argument type mismatch in printf for format specifier 's' at argument index 1.
; WIDE_STRING-NEXT: error: Execution of function 'wide_string' failed.
