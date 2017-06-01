; RUN: opt -S -instcombine < %s | FileCheck %s

; https://llvm.org/bugs/show_bug.cgi?id=25900
; An arithmetic shift right of a power of two is not a power
; of two if the original value is the sign bit. Therefore,
; we can't transform the sdiv into a udiv.

define i32 @pr25900(i32 %d) {
  %and = and i32 %d, -2147483648
; The next 3 lines prevent another fold from masking the bug.
  %ext = zext i32 %and to i64
  %or = or i64 %ext, 4294967296
  %trunc = trunc i64 %or to i32
  %ashr = ashr exact i32 %trunc, 31
  %div = sdiv i32 4, %ashr
  ret i32 %div

; CHECK: sdiv
}

