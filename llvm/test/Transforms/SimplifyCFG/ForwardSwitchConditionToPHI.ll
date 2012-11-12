; RUN: opt < %s -simplifycfg -S | \
; RUN:   not grep " switch"
; PR10131

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

define i32 @t(i32 %m) nounwind readnone {
entry:
  switch i32 %m, label %sw.bb4 [
    i32 0, label %sw.bb0
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb0:                                           ; preds = %entry
  br label %return

sw.bb1:                                           ; preds = %entry
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

sw.bb4:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1
  %retval.0 = phi i32 [ 4, %sw.bb4 ], [ 3, %sw.bb3 ], [ 2, %sw.bb2 ], [ 1, %sw.bb1 ], [ 0, %sw.bb0 ]
  ret i32 %retval.0
}
