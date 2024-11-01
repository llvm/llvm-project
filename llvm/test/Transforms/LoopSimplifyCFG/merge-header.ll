; RUN: opt -S -passes='require<domtree>,loop(loop-simplifycfg)' < %s | FileCheck %s
; RUN: opt -S -passes=loop-simplifycfg -verify-memoryssa < %s | FileCheck %s

; CHECK-LABEL: foo
; CHECK:      entry:
; CHECK-NEXT:   br label %[[LOOP:[a-z]+]]
; CHECK:      [[LOOP]]:
; CHECK-NEXT:   phi
; CHECK-NOT:    br label
; CHECK:        br i1
define i32 @foo(i32* %P, i64* %Q) {
entry:
  br label %outer

outer:                                            ; preds = %outer.latch2, %entry
  %y.2 = phi i32 [ 0, %entry ], [ %y.inc2, %outer.latch2 ]
  br label %inner

inner:                                            ; preds = %outer
  store i32 0, i32* %P
  store i32 1, i32* %P
  store i32 2, i32* %P
  %y.inc2 = add nsw i32 %y.2, 1
  %exitcond.outer = icmp eq i32 %y.inc2, 3
  store i32 %y.2, i32* %P
  br i1 %exitcond.outer, label %exit, label %outer.latch2

outer.latch2:                                     ; preds = %inner
  %t = sext i32 %y.inc2 to i64
  store i64 %t, i64* %Q
  br label %outer

exit:                                             ; preds = %inner
  ret i32 0
}
