; RUN: opt -mergefunc -disable-output < %s
; PR15185
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

%.qux.2496 = type { i32, %.qux.2497 }
%.qux.2497 = type { i8, i32 }
%.qux.2585 = type { i32, i32, i8* }

@g2 = external unnamed_addr constant [9 x i8], align 1
@g3 = internal unnamed_addr constant [1 x i8*] [i8* bitcast (i8* (%.qux.2585*)* @func35 to i8*)]

define internal i32 @func1(i32* %ptr, { i32, i32 }* nocapture %method) align 2 {
  br label %1

; <label>:1
  br label %2

; <label>:2
  ret i32 undef
}

define internal i32 @func10(%.qux.2496* nocapture %this) align 2 {
  %1 = getelementptr inbounds %.qux.2496, %.qux.2496* %this, i32 0, i32 1, i32 1
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}

define internal i8* @func29(i32* nocapture %this) align 2 {
  ret i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g2, i32 0, i32 0)
}

define internal i32* @func33(%.qux.2585* nocapture %this) align 2 {
  ret i32* undef
}

define internal i32* @func34(%.qux.2585* nocapture %this) align 2 {
  %1 = getelementptr inbounds %.qux.2585, %.qux.2585* %this, i32 0
  ret i32* undef
}

define internal i8* @func35(%.qux.2585* nocapture %this) align 2 {
  %1 = getelementptr inbounds %.qux.2585, %.qux.2585* %this, i32 0, i32 2
  %2 = load i8*, i8** %1, align 4
  ret i8* %2
}
