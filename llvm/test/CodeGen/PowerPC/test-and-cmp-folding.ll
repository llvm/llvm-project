; RUN: llc < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 \
; RUN:   -verify-machineinstrs | FileCheck %s
declare void @free()

define dso_local fastcc void @test2() {
; CHECK-LABEL: test2
entry:
  switch i16 undef, label %sw.default [
    i16 10, label %sw.bb52
    i16 134, label %sw.bb54
  ]

sw.default:                                       ; preds = %entry
  unreachable


sw.bb52:                                          ; preds = %entry, %entry, %entry, %entry, %entry, %entry
  br i1 undef, label %if.then14.i, label %sw.epilog.i642

if.then14.i:                                      ; preds = %sw.bb52
  %call39.i = call i64 @test() #3
  %and.i126.i = and i64 %call39.i, 1
  br i1 undef, label %dummy.exit.i, label %if.then.i.i.i636

if.then.i.i.i636:                                 ; preds = %if.then14.i
  %0 = load i8*, i8** undef, align 8
  call void @free() #3
  br label %dummy.exit.i

dummy.exit.i:               ; preds = %if.then.i.i.i636, %if.then14.i
; CHECK: # %dummy.exit.i
; CHECK-NEXT: andi.
; CHECK-NEXT: bc 12
  %cond82.i = icmp eq i64 %and.i126.i, 0
  br i1 %cond82.i, label %if.end50.i, label %dummy.exit

if.end50.i:                                       ; preds = %dummy.exit.i
  unreachable

sw.epilog.i642:                                   ; preds = %sw.bb52
  unreachable

dummy.exit: ; preds = %dummy.exit.i
  unreachable

sw.bb54:                                          ; preds = %entry, %entry
  call fastcc void @test3()
  unreachable
}

declare dso_local fastcc void @test3()

declare i64 @test()
