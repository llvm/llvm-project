; RUN: opt -codegenprepare  < %s  -mtriple=aarch64-none-linux-gnu -S  | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; Expect to skip merging two empty blocks (sw.bb and sw.bb2) into sw.epilog
; as both of them are unlikely executed.
define i32 @f_switch(i32 %c)  {
; CHECK-LABEL: @f_switch
; CHECK-LABEL: entry:
; CHECK: i32 10, label %sw.bb
; CHECK: i32 20, label %sw.bb2
entry:
  switch i32 %c, label %sw.default [
    i32 10, label %sw.bb
    i32 20, label %sw.bb2
    i32 30, label %sw.bb3
    i32 40, label %sw.bb4
  ], !prof !0

sw.bb:                                            ; preds = %entry
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase3 to void ()*)()
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase4 to void ()*)()
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  call void bitcast (void (...)* @calldefault to void ()*)()
  br label %sw.epilog

; CHECK-LABEL: sw.epilog:
; CHECK: %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F2, %sw.bb2 ], [ @F1, %sw.bb ]
sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb
  %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F2, %sw.bb2 ], [ @F1, %sw.bb ]
  %callee.knr.cast = bitcast void (...)* %fp.0 to void ()*
  call void %callee.knr.cast()
  ret i32 0
}

; Expect not to merge sw.bb2 because of the conflict in the incoming value from
; sw.bb which is already merged.
define i32 @f_switch2(i32 %c)  {
; CHECK-LABEL: @f_switch2
; CHECK-LABEL: entry:
; CHECK: i32 10, label %sw.epilog
; CHECK: i32 20, label %sw.bb2
entry:
  switch i32 %c, label %sw.default [
    i32 10, label %sw.bb
    i32 20, label %sw.bb2
    i32 30, label %sw.bb3
    i32 40, label %sw.bb4
  ], !prof !1

sw.bb:                                            ; preds = %entry
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase3 to void ()*)()
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase4 to void ()*)()
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  call void bitcast (void (...)* @calldefault to void ()*)()
  br label %sw.epilog

; CHECK-LABEL: sw.epilog:
; CHECK: %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F2, %sw.bb2 ], [ @F1, %entry ]
sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb
  %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F2, %sw.bb2 ], [ @F1, %sw.bb ]
  %callee.knr.cast = bitcast void (...)* %fp.0 to void ()*
  call void %callee.knr.cast()
  ret i32 0
}

; Multiple empty blocks should be considered together if all incoming values
; from them are same.  We expect to merge both empty blocks (sw.bb and sw.bb2)
; because the sum of frequencies are higer than the threshold.
define i32 @f_switch3(i32 %c)  {
; CHECK-LABEL: @f_switch3
; CHECK-LABEL: entry:
; CHECK: i32 10, label %sw.epilog
; CHECK: i32 20, label %sw.epilog
entry:
  switch i32 %c, label %sw.default [
    i32 10, label %sw.bb
    i32 20, label %sw.bb2
    i32 30, label %sw.bb3
    i32 40, label %sw.bb4
  ], !prof !2

sw.bb:                                            ; preds = %entry
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase3 to void ()*)()
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  call void bitcast (void (...)* @callcase4 to void ()*)()
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  call void bitcast (void (...)* @calldefault to void ()*)()
  br label %sw.epilog

; CHECK-LABEL: sw.epilog:
; CHECK: %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F1, %entry ], [ @F1, %entry ]
sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb
  %fp.0 = phi void (...)* [ @FD, %sw.default ], [ @F4, %sw.bb4 ], [ @F3, %sw.bb3 ], [ @F1, %sw.bb2 ], [ @F1, %sw.bb ]
  %callee.knr.cast = bitcast void (...)* %fp.0 to void ()*
  call void %callee.knr.cast()
  ret i32 0
}

declare void @F1(...) local_unnamed_addr
declare void @F2(...) local_unnamed_addr
declare void @F3(...) local_unnamed_addr
declare void @F4(...) local_unnamed_addr
declare void @FD(...) local_unnamed_addr
declare void @callcase3(...) local_unnamed_addr
declare void @callcase4(...) local_unnamed_addr
declare void @calldefault(...) local_unnamed_addr

!0 = !{!"branch_weights", i32 5, i32 1, i32 1,i32 5, i32 5}
!1 = !{!"branch_weights", i32 1 , i32 5, i32 1,i32 1, i32 1}
!2 = !{!"branch_weights", i32 1 , i32 4, i32 1,i32 1, i32 1}
