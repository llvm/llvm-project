; RUN: llc -relocation-model=static < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

define i32 @t1(i32 %a, i32 %b) nounwind {
entry:
; ELF64: t1
  %x = add i32 %a, %b  
  br i1 1, label %if.then, label %if.else
; ELF64-NOT: b {{\.?}}LBB0_1

if.then:                                          ; preds = %entry
  call void @foo1()
  br label %if.end7

if.else:                                          ; preds = %entry
  br i1 0, label %if.then2, label %if.else3
; ELF64: b {{\.?}}LBB0_4

if.then2:                                         ; preds = %if.else
  call void @foo2()
  br label %if.end6

if.else3:                                         ; preds = %if.else
  %y = sub i32 %a, %b
  br i1 1, label %if.then5, label %if.end
; ELF64-NOT: b {{\.?}}LBB0_5

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
