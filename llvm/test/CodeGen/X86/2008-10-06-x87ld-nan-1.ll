; ModuleID = 'nan.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-f80:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
; RUN: llc < %s -mattr=-sse2,-sse3,-sse | grep fldl
; This NaN should be shortened to a double (not a float).

declare x86_stdcallcc void @_D3nan5printFeZv(x86_fp80 %f)

define i32 @main() {
entry_nan.main:
  call x86_stdcallcc void @_D3nan5printFeZv(x86_fp80 0xK7FFFC001234000000800)
  ret i32 0
}
