; RUN: opt -inline %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; Check we don't inline loops at -Oz. They tend to be larger than we
; expect.

; CHECK: define i8* @H
@digits = constant [16 x i8] c"0123456789ABCDEF", align 1
define i8* @H(i8* %p, i32 %val, i32 %num) #0 {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %p.addr.0 = phi i8* [ %p, %entry ], [ %incdec.ptr, %do.body ]
  %val.addr.0 = phi i32 [ %val, %entry ], [ %shl, %do.body ]
  %num.addr.0 = phi i32 [ %num, %entry ], [ %dec, %do.body ]
  %shr = lshr i32 %val.addr.0, 28
  %arrayidx = getelementptr inbounds [16 x i8], [16 x i8]* @digits, i32 0, i32 %shr
  %0 = load i8, i8* %arrayidx, align 1
  %incdec.ptr = getelementptr inbounds i8, i8* %p.addr.0, i32 1
  store i8 %0, i8* %p.addr.0, align 1
  %shl = shl i32 %val.addr.0, 4
  %dec = add i32 %num.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  %scevgep = getelementptr i8, i8* %p, i32 %num
  ret i8* %scevgep
}

define nonnull i8* @call1(i8* %p, i32 %val, i32 %num) #0 {
entry:
; CHECK: tail call i8* @H
  %call = tail call i8* @H(i8* %p, i32 %val, i32 %num) #0
  ret i8* %call
}

define nonnull i8* @call2(i8* %p, i32 %val) #0 {
entry:
; CHECK: tail call i8* @H
  %call = tail call i8* @H(i8* %p, i32 %val, i32 32) #0
  ret i8* %call
}

attributes #0 = { minsize optsize }

