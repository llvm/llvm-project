; RUN: llc < %s | grep 0x0021 | count 2
; PR4776
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-unknown-unknown"

@"\010x0021" = external global i8, align 1        ; <i8*> [#uses=2]

define zeroext i8 @foo(i8 zeroext %x) nounwind {
entry:
  %retval = alloca i8                             ; <i8*> [#uses=2]
  %x.addr = alloca i8                             ; <i8*> [#uses=2]
  %tmp = alloca i8, align 1                       ; <i8*> [#uses=2]
  store i8 %x, ptr %x.addr
  %tmp1 = load volatile i8, ptr @"\010x0021"          ; <i8> [#uses=1]
  store i8 %tmp1, ptr %tmp
  %tmp2 = load i8, ptr %x.addr                        ; <i8> [#uses=1]
  store volatile i8 %tmp2, ptr @"\010x0021"
  %tmp3 = load i8, ptr %tmp                           ; <i8> [#uses=1]
  store i8 %tmp3, ptr %retval
  %0 = load i8, ptr %retval                           ; <i8> [#uses=1]
  ret i8 %0
}
