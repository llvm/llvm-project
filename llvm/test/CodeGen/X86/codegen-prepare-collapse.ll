; RUN: llc -fast-isel=true -O1 -mtriple=x86_64-unknown-linux-gnu -start-before=codegenprepare -stop-after=codegenprepare -o - < %s | FileCheck %s

; CHECK-LABEL: @foo
define void @foo() {
top:
; CHECK: br label %L34
  br label %L34

L34:                                              ; preds = %L34, %L34, %top
  %.sroa.075.0 = phi i64 [ undef, %top ], [ undef, %L34 ], [ undef, %L34 ]
  %0 = icmp sgt i8 undef, -1
  %cond5896 = icmp eq i8 0, 2
  %cond58 = and i1 %cond5896, %0
; During codegenprepare such degenerate branches can occur and should not
; lead to crashes.
; CHECK: br label %L34
  br i1 %cond58, label %L34, label %L34
}
