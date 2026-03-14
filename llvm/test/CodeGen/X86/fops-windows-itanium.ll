; RUN: llc -mtriple i686-windows-itanium -filetype asm -o - %s | FileCheck %s

declare float @llvm.ceil.f32(float)
declare float @llvm.cos.f32(float)
declare float @llvm.exp.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.log.f32(float)
declare float @llvm.log10.f32(float)
declare float @llvm.pow.f32(float, float)
declare float @llvm.sin.f32(float)

define float @f(float %f) {
  %r = call float @llvm.ceil.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _f:
; CHECK-NOT: calll _ceilf
; CHECK: calll _ceil{{$}}

define float @g(float %f) {
  %r = call float @llvm.cos.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _g:
; CHECK-NOT: calll _cosf
; CHECK: calll _cos{{$}}

define float @h(float %f) {
  %r = call float @llvm.exp.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _h:
; CHECK-NOT: calll _expf
; CHECK: calll _exp{{$}}

define float @i(float %f) {
  %r = call float @llvm.floor.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _i:
; CHECK-NOT: calll _floorf
; CHECK: calll _floor{{$}}

define float @j(float %f, float %g) {
  %r =  frem float %f, %g
  ret float %r
}

; CHECK-LABEL: _j:
; CHECK-NOT: calll _fmodf
; CHECK: calll _fmod{{$}}

define float @k(float %f) {
  %r = call float @llvm.log.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _k:
; CHECK-NOT: calll _logf
; CHECK: calll _log{{$}}

define float @l(float %f) {
  %r = call float @llvm.log10.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _l:
; CHECK-NOT: calll _log10f
; CHECK: calll _log10{{$}}

define float @m(float %f, float %g) {
  %r = call float @llvm.pow.f32(float %f, float %g)
  ret float %r
}

; CHECK-LABEL: _m:
; CHECK-NOT: calll _powf
; CHECK: calll _pow{{$}}

define float @n(float %f) {
  %r = call float @llvm.sin.f32(float %f)
  ret float %r
}

; CHECK-LABEL: _n:
; CHECK-NOT: calll _sinf
; CHECK: calll _sin{{$}}

