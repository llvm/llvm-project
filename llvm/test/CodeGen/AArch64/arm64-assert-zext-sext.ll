; RUN: llc -O2 -mtriple=aarch64-linux-gnu < %s | FileCheck %s

declare i32 @test(i32) local_unnamed_addr
declare i32 @test1(i64) local_unnamed_addr

define i32 @assertzext(i32 %n, i1 %a, i32* %b) local_unnamed_addr {
entry:
  %i = select i1 %a, i64 0, i64 66296709418
  %conv.i = trunc i64 %i to i32
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                     ; preds = %entry
  store i32 0, i32* %b, align 4
  br label %if.end

if.end:                      ; preds = %if.then, %entry
  %i1 = phi i32 [ 0, %if.then ], [ %conv.i, %entry ]
  %call.i = tail call i32 @test(i32 %i1)
  %i2 = sext i32 %i1 to i64
  %call1.i = tail call i32 @test1(i64 %i2)
  ret i32 0
; CHECK:       // %if.end
; CHECK:       mov     w{{[0-9]+}}, w{{[0-9]+}}
; CHECK:       bl      test
; CHECK:       mov     w{{[0-9]+}}, w{{[0-9]+}}
; CHECK:       bl      test1
}

define i32 @assertsext(i32 %n, i8 %a) local_unnamed_addr {
entry:
  %conv.i = sext i8 %a to i32
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                     ; preds = %entry
  %conv1 = zext i32 %conv.i to i64
  %div = udiv i64 2036854775807, %conv1
  br label %if.end
; CHECK:       // %if.then
; CHECK:       mov     w{{[0-9]+}}, w{{[0-9]+}}
; CHECK:       udiv    x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}

if.end:                      ; preds = %if.then, %entry
  %i1 = phi i64 [ %div, %if.then ], [ 0, %entry ]
  %call1.i = tail call i32 @test1(i64 %i1)
  ret i32 0
}
