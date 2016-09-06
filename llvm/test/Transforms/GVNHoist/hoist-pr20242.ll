; RUN: opt -gvn-hoist -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that all "or" expressions are hoisted.
; CHECK-LABEL: @encode
; CHECK: or i32
; CHECK-NOT: or i32

define i8* @encode(i8* %p, i32 %v) {
entry:
  %p.addr = alloca i8*, align 8
  %v.addr = alloca i32, align 4
  store i8* %p, i8** %p.addr, align 8
  store i32 %v, i32* %v.addr, align 4
  %0 = load i32, i32* %v.addr, align 4
  %cmp = icmp ult i32 %0, 23
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %v.addr, align 4
  %or = or i32 %1, 128
  %conv = trunc i32 %or to i8
  %2 = load i8*, i8** %p.addr, align 8
  %incdec.ptr = getelementptr inbounds i8, i8* %2, i32 1
  store i8* %incdec.ptr, i8** %p.addr, align 8
  store i8 %conv, i8* %2, align 1
  br label %if.end15

if.else:                                          ; preds = %entry
  %3 = load i32, i32* %v.addr, align 4
  %cmp1 = icmp ult i32 %3, 42
  br i1 %cmp1, label %if.then3, label %if.else9

if.then3:                                         ; preds = %if.else
  %4 = load i32, i32* %v.addr, align 4
  %or4 = or i32 %4, 128
  %conv5 = trunc i32 %or4 to i8
  %5 = load i8*, i8** %p.addr, align 8
  %incdec.ptr6 = getelementptr inbounds i8, i8* %5, i32 1
  store i8* %incdec.ptr6, i8** %p.addr, align 8
  store i8 %conv5, i8* %5, align 1
  %6 = load i32, i32* %v.addr, align 4
  %conv7 = trunc i32 %6 to i8
  %7 = load i8*, i8** %p.addr, align 8
  %incdec.ptr8 = getelementptr inbounds i8, i8* %7, i32 1
  store i8* %incdec.ptr8, i8** %p.addr, align 8
  store i8 %conv7, i8* %7, align 1
  br label %if.end

if.else9:                                         ; preds = %if.else
  %8 = load i32, i32* %v.addr, align 4
  %or10 = or i32 %8, 128
  %conv11 = trunc i32 %or10 to i8
  %9 = load i8*, i8** %p.addr, align 8
  %incdec.ptr12 = getelementptr inbounds i8, i8* %9, i32 1
  store i8* %incdec.ptr12, i8** %p.addr, align 8
  store i8 %conv11, i8* %9, align 1
  %10 = load i32, i32* %v.addr, align 4
  %shr = lshr i32 %10, 7
  %conv13 = trunc i32 %shr to i8
  %11 = load i8*, i8** %p.addr, align 8
  %incdec.ptr14 = getelementptr inbounds i8, i8* %11, i32 1
  store i8* %incdec.ptr14, i8** %p.addr, align 8
  store i8 %conv13, i8* %11, align 1
  br label %if.end

if.end:                                           ; preds = %if.else9, %if.then3
  br label %if.end15

if.end15:                                         ; preds = %if.end, %if.then
  %12 = load i8*, i8** %p.addr, align 8
  ret i8* %12
}
