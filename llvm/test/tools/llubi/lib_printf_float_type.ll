; RUN: llubi < %s 2>&1 | FileCheck %s --check-prefix=DOUBLE
; RUN: not llubi --entry-function=half_arg < %s 2>&1 | FileCheck %s --check-prefix=HALF
; RUN: not llubi --entry-function=bfloat_arg < %s 2>&1 | FileCheck %s --check-prefix=BFLOAT
; RUN: not llubi --entry-function=float_arg < %s 2>&1 | FileCheck %s --check-prefix=FLOAT
; RUN: not llubi --entry-function=fp80_arg < %s 2>&1 | FileCheck %s --check-prefix=FP80
; RUN: not llubi --entry-function=fp128_arg < %s 2>&1 | FileCheck %s --check-prefix=FP128
; RUN: not llubi --entry-function=ppc_fp128_arg < %s 2>&1 | FileCheck %s --check-prefix=PPC-FP128
; RUN: not llubi --entry-function=float_upper_hex_arg < %s 2>&1 | FileCheck %s --check-prefix=FLOAT-UPPER-HEX

@fmt_all = constant [22 x i8] c"%f %e %E %g %G %a %A\0A\00"
@fmt_f = constant [3 x i8] c"%f\00"
@fmt_e = constant [3 x i8] c"%e\00"
@fmt_E = constant [3 x i8] c"%E\00"
@fmt_g = constant [3 x i8] c"%g\00"
@fmt_G = constant [3 x i8] c"%G\00"
@fmt_a = constant [3 x i8] c"%a\00"
@fmt_A = constant [3 x i8] c"%A\00"

declare i32 @printf(ptr, ...)

define void @main() {
  call i32 (ptr, ...) @printf(ptr @fmt_all, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0, double 1.0)
  ret void
}

; DOUBLE: 1.000000 1.000000e+00 1.000000E+00 1 1 {{0x1(\.0+)?p\+0+}} {{0X1(\.0+)?P\+0+}}

define void @half_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_f, half 0.0)
  ret void
}

; HALF: Immediate UB detected: Argument type mismatch in printf for format specifier 'f' at argument index 1.
; HALF-NEXT: error: Execution of function 'half_arg' failed.

define void @bfloat_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_e, bfloat 0.0)
  ret void
}

; BFLOAT: Immediate UB detected: Argument type mismatch in printf for format specifier 'e' at argument index 1.
; BFLOAT-NEXT: error: Execution of function 'bfloat_arg' failed.

define void @float_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_E, float 0.0)
  ret void
}

; FLOAT: Immediate UB detected: Argument type mismatch in printf for format specifier 'E' at argument index 1.
; FLOAT-NEXT: error: Execution of function 'float_arg' failed.

define void @fp80_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_g, x86_fp80 0xK00000000000000000000)
  ret void
}

; FP80: Immediate UB detected: Argument type mismatch in printf for format specifier 'g' at argument index 1.
; FP80-NEXT: error: Execution of function 'fp80_arg' failed.

define void @fp128_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_G, fp128 0xL00000000000000000000000000000000)
  ret void
}

; FP128: Immediate UB detected: Argument type mismatch in printf for format specifier 'G' at argument index 1.
; FP128-NEXT: error: Execution of function 'fp128_arg' failed.

define void @ppc_fp128_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_a, ppc_fp128 0xM00000000000000000000000000000000)
  ret void
}

; PPC-FP128: Immediate UB detected: Argument type mismatch in printf for format specifier 'a' at argument index 1.
; PPC-FP128-NEXT: error: Execution of function 'ppc_fp128_arg' failed.

define void @float_upper_hex_arg() {
  call i32 (ptr, ...) @printf(ptr @fmt_A, float 0.0)
  ret void
}

; FLOAT-UPPER-HEX: Immediate UB detected: Argument type mismatch in printf for format specifier 'A' at argument index 1.
; FLOAT-UPPER-HEX-NEXT: error: Execution of function 'float_upper_hex_arg' failed.
