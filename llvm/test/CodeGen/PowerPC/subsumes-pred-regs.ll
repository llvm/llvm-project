; RUN: llc < %s -mcpu=ppc64 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define zeroext i1 @test1() unnamed_addr #0 align 2 {

; CHECK-LABEL: @test1

entry:
  br i1 undef, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  unreachable

lor.end:                                          ; preds = %entry
  br i1 undef, label %land.rhs, label %if.then

if.then:                                          ; preds = %lor.end
  br i1 undef, label %return, label %if.end.i24

if.end.i24:                                       ; preds = %if.then
  %0 = load i32* undef, align 4
  %lnot.i.i16.i23 = icmp eq i32 %0, 0
  br i1 %lnot.i.i16.i23, label %if.end7.i37, label %test.exit27.i34

test.exit27.i34: ; preds = %if.end.i24
  br i1 undef, label %return, label %if.end7.i37

if.end7.i37:                                      ; preds = %test.exit27.i34, %if.end.i24
  %tobool.i.i36 = icmp eq i8 undef, 0
  br i1 %tobool.i.i36, label %return, label %if.then9.i39

if.then9.i39:                                     ; preds = %if.end7.i37
  br i1 %lnot.i.i16.i23, label %return, label %lor.rhs.i.i49

; CHECK: .LBB0_7:
; CHECK:	beq 1, .LBB0_10
; CHECK:	beq 0, .LBB0_10
; CHECK: .LBB0_9:

lor.rhs.i.i49:                                    ; preds = %if.then9.i39
  %cmp.i.i.i.i48 = icmp ne i64 undef, 0
  br label %return

land.rhs:                                         ; preds = %lor.end
  br i1 undef, label %return, label %if.end.i

if.end.i:                                         ; preds = %land.rhs
  br i1 undef, label %return, label %if.then9.i

if.then9.i:                                       ; preds = %if.end.i
  br i1 undef, label %return, label %lor.rhs.i.i

lor.rhs.i.i:                                      ; preds = %if.then9.i
  %cmp.i.i.i.i = icmp ne i64 undef, 0
  br label %return

return:                                           ; preds = %lor.rhs.i.i, %if.then9.i, %if.end.i, %land.rhs, %lor.rhs.i.i49, %if.then9.i39, %if.end7.i37, %test.exit27.i34, %if.then
  %retval.0 = phi i1 [ false, %if.then ], [ false, %test.exit27.i34 ], [ true, %if.end7.i37 ], [ true, %if.then9.i39 ], [ %cmp.i.i.i.i48, %lor.rhs.i.i49 ], [ false, %land.rhs ], [ true, %if.end.i ], [ true, %if.then9.i ], [ %cmp.i.i.i.i, %lor.rhs.i.i ]
  ret i1 %retval.0
}

attributes #0 = { nounwind }

