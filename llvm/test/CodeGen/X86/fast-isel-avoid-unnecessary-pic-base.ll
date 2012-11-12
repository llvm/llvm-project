; RUN: llc -O0 -relocation-model=pic < %s | not grep call
; rdar://8396318

; Don't emit a PIC base register if no addresses are needed.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

define i32 @foo(i32 %x, i32 %y, i32 %z) nounwind ssp {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  store i32 %z, i32* %z.addr, align 4
  %tmp = load i32* %x.addr, align 4
  %tmp1 = load i32* %y.addr, align 4
  %add = add nsw i32 %tmp, %tmp1
  %tmp2 = load i32* %z.addr, align 4
  %add3 = add nsw i32 %add, %tmp2
  ret i32 %add3
}
