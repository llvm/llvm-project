; RUN: llc -mtriple=armv7-linux < %s -o - | FileCheck -check-prefix=LINUX %s
; RUN: llc -mtriple=thumbv7-windows-msvc -mattr=+thumb-mode < %s -o - | FileCheck -check-prefix=WINDOWS %s

define double @testExp(double %val, i32 %a) {
; LINUX:    b ldexp{{$}}
; WINDOWS:  b.w ldexp{{$}}
entry:
  %call = tail call fast double @ldexp(double %val, i32 %a)
  ret double %call
}

declare double @ldexp(double, i32) memory(none)

define double @testExpIntrinsic(double %val, i32 %a) {
; LINUX:    b ldexp{{$}}
; WINDOWS:  b.w ldexp{{$}}
entry:
  %call = tail call fast double @llvm.ldexp.f64(double %val, i32 %a)
  ret double %call
}

define float @testExpf(float %val, i32 %a) {
; LINUX:    b ldexpf
; WINDOWS:  b.w ldexpf
entry:
  %call = tail call fast float @ldexpf(float %val, i32 %a)
  ret float %call
}

define float @testExpfIntrinsic(float %val, i32 %a) {
; LINUX:    b ldexpf
; WINDOWS:  bl ldexp{{$}}
entry:
  %call = tail call fast float @llvm.ldexp.f32(float %val, i32 %a)
  ret float %call
}

declare float @ldexpf(float, i32) memory(none)

define fp128 @testExpl(fp128 %val, i32 %a) {
; LINUX:    bl ldexpl
; WINDOWS:    b.w ldexpl
entry:
  %call = tail call fast fp128 @ldexpl(fp128 %val, i32 %a)
  ret fp128 %call
}

declare fp128 @ldexpl(fp128, i32) memory(none)

define half @testExpf16(half %val, i32 %a) {
; LINUX: bl ldexpf
; WINDOWS: bl ldexp{{$}}
entry:
  %0 = tail call fast half @llvm.ldexp.f16.i32(half %val, i32 %a)
  ret half %0
}

declare half @llvm.ldexp.f16.i32(half, i32) memory(none)
