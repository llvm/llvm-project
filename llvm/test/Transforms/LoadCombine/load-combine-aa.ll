; RUN: opt -basicaa -load-combine -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @test1(i32* nocapture readonly noalias %a, i32* nocapture readonly noalias %b) {
; CHECK-LABEL: @test1

; CHECK: load i64, i64*
; CHECK: ret i64

  %load1 = load i32, i32* %a, align 4
  %conv = zext i32 %load1 to i64
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %load1, i32* %b, align 4
  %load2 = load i32, i32* %arrayidx1, align 4
  %conv2 = zext i32 %load2 to i64
  %shl = shl nuw i64 %conv2, 32
  %add = or i64 %shl, %conv
  ret i64 %add
}

define i64 @test2(i32* nocapture readonly %a, i32* nocapture readonly %b) {
; CHECK-LABEL: @test2

; CHECK-NOT: load i64
; CHECK: load i32, i32*
; CHECK: load i32, i32*
; CHECK: ret i64

  %load1 = load i32, i32* %a, align 4
  %conv = zext i32 %load1 to i64
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %load1, i32* %b, align 4
  %load2 = load i32, i32* %arrayidx1, align 4
  %conv2 = zext i32 %load2 to i64
  %shl = shl nuw i64 %conv2, 32
  %add = or i64 %shl, %conv
  ret i64 %add
}

%rec11 = type { i16, i16, i16 }
@str = global %rec11 { i16 1, i16 2, i16 3 }

; PR31517 - Check that loads which span an aliasing store are not combined.
define i16 @test3() {
; CHECK-LABEL: @test3

; CHECK-NOT: load i32
; CHECK: load i16, i16*
; CHECK: store i16
; CHECK: load i16, i16*
; CHECK: ret i16

  %_tmp9 = getelementptr %rec11, %rec11* @str, i16 0, i32 1
  %_tmp10 = load i16, i16* %_tmp9
  %_tmp12 = getelementptr %rec11, %rec11* @str, i16 0, i32 0
  store i16 %_tmp10, i16* %_tmp12
  %_tmp13 = getelementptr %rec11, %rec11* @str, i16 0, i32 0
  %_tmp14 = load i16, i16* %_tmp13
  %_tmp15 = icmp eq i16 %_tmp14, 3
  %_tmp16 = select i1 %_tmp15, i16 1, i16 0
  ret i16 %_tmp16
}
