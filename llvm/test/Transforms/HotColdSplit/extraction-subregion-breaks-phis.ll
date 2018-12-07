; RUN: opt -S -hotcoldsplit < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: call {{.*}}@foo.cold.1(
; CHECK: unreachable

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: switch i32 undef, label %sw.epilog.i
define void @foo(i32 %QMM) {
entry:
  switch i32 %QMM, label %entry.if.end16_crit_edge [
    i32 1, label %if.then
  ]

entry.if.end16_crit_edge:                         ; preds = %entry
  br label %if.end16

if.then:                                          ; preds = %entry
  br i1 undef, label %cond.true.i.i, label %_ZN10StringView8popFrontEv.exit.i

cond.true.i.i:                                    ; preds = %if.then
  ret void

_ZN10StringView8popFrontEv.exit.i:                ; preds = %if.then
  switch i32 undef, label %sw.epilog.i [
    i32 81, label %if.end16
    i32 82, label %sw.bb4.i
    i32 83, label %sw.bb8.i
    i32 84, label %sw.bb12.i
    i32 65, label %if.end16
    i32 66, label %sw.bb20.i
    i32 67, label %sw.bb24.i
    i32 68, label %sw.bb28.i
  ]

sw.bb4.i:                                         ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.bb8.i:                                         ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.bb12.i:                                        ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.bb20.i:                                        ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.bb24.i:                                        ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.bb28.i:                                        ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

sw.epilog.i:                                      ; preds = %_ZN10StringView8popFrontEv.exit.i
  br label %if.end16

if.end16:                                         ; preds = %sw.epilog.i, %sw.bb28.i, %sw.bb24.i, %sw.bb20.i, %sw.bb12.i, %sw.bb8.i, %sw.bb4.i, %_ZN10StringView8popFrontEv.exit.i, %_ZN10StringView8popFrontEv.exit.i, %entry.if.end16_crit_edge
  %0 = phi i8 [ 0, %entry.if.end16_crit_edge ], [ 0, %_ZN10StringView8popFrontEv.exit.i ], [ 0, %_ZN10StringView8popFrontEv.exit.i ], [ 1, %sw.bb4.i ], [ 2, %sw.bb8.i ], [ 3, %sw.bb12.i ], [ 1, %sw.bb20.i ], [ 2, %sw.bb24.i ], [ 3, %sw.bb28.i ], [ 0, %sw.epilog.i ]
  unreachable
}
