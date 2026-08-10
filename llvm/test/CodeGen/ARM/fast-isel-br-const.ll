; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @t1(i32 %a, i32 %b) nounwind uwtable ssp {
entry:
; THUMB-LABEL: t1:
; ARM-LABEL: t1:
  %x = add i32 %a, %b  
  br i1 1, label %if.then, label %if.else
; THUMB-NOT: b {{\.?}}LBB0_1
; ARM-NOT:  b {{\.?}}LBB0_1

if.then:                                          ; preds = %entry
  call void @foo1()
  br label %if.end7

if.else:                                          ; preds = %entry
  br i1 0, label %if.then2, label %if.else3
; THUMB: b {{\.?}}LBB0_4
; ARM:  b {{\.?}}LBB0_4

if.then2:                                         ; preds = %if.else
  call void @foo2()
  br label %if.end6

if.else3:                                         ; preds = %if.else
  %y = sub i32 %a, %b
  br i1 1, label %if.then5, label %if.end
; THUMB-NOT: b {{\.?}}LBB0_5
; ARM-NOT:  b {{\.?}}LBB0_5

if.then5:                                         ; preds = %if.else3
  call void @foo1()
  br label %if.end

if.end:                                           ; preds = %if.then5, %if.else3
  br label %if.end6

if.end6:                                          ; preds = %if.end, %if.then2
  br label %if.end7

if.end7:                                          ; preds = %if.end6, %if.then
  ret i32 0
}

declare void @foo1()

declare void @foo2()
