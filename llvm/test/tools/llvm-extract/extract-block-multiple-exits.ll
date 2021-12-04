; RUN: llvm-extract -S -bb "foo:region_start;exiting0;exiting1" %s --bb-keep-functions --bb-keep-blocks | FileCheck %s


; CHECK-LABEL: define void @foo(
;
; CHECK:       outsideonly:
; CHECK-NEXT:    store i32 0, i32* %arg, align 4
; CHECK-NEXT:    br label %cleanup
;
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @foo.region_start(i32* %arg)
; CHECK-NEXT:    br label %return
;
; CHECK:       extractonly:
; CHECK-NEXT:    store i32 1, i32* %arg, align 4
; CHECK-NEXT:    br label %cleanup
;
; CHECK:       cleanup:
; CHECK-NEXT:    %dest = phi i8 [ 0, %outsideonly ], [ 1, %extractonly ]
; CHECK-NEXT:    switch


; CHECK-LABEL: define internal void @foo.region_start(i32* %arg) {
; CHECK:         br label %region_start
;
; CHECK:      region_start:
; CHECK-NEXT:    br label %extractonly
; CHECK-EMPTY:
; CHECK-NEXT:  extractonly:
; CHECK-NEXT:    store i32 1, i32* %arg, align 4
; CHECK-NEXT:    br label %cleanup
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup:
; CHECK-NEXT:    %dest = phi i8 [ 1, %extractonly ]
; CHECK-NEXT:    switch i8 %dest, label %fallback [
; CHECK-NEXT:      i8 0, label %return.exitStub
; CHECK-NEXT:      i8 1, label %region_end
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  fallback:
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  region_end:
; CHECK-NEXT:    br label %return.exitStub
; CHECK-EMPTY:
; CHECK-NEXT:  return.exitStub:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }


define void @foo(i32* %arg, i1 %c0, i1 %c1, i1 %c2, i8 %dest) {
entry:
  br i1 %c0, label %region_start, label %exit

region_start:
  %a = add i32 42, 1
  br i1 %c1, label %exiting0, label %exiting1

exiting0:
  %b = add i32 42, 2
  br i1 %c2, label %exit, label %exit0

exiting1:
  %c = add i32 42, 3
  switch i8 %dest, label %fallback [
    i8 0, label %exit0
    i8 1, label %exit1
    i8 2, label %exit2
    i8 3, label %exit0
  ]

fallback:
  unreachable

exit:
  %A = phi i32 [ 42, %entry ], [ %b, %exiting0 ]
  store i32 %A, i32* %arg
  br label %return

exit0:
  %B = phi i32 [ %b, %exiting0 ], [ %a, %exiting1 ] , [ %a, %exiting1 ] ; Not working without --bb-keep-blocks (different incoming value after %exiting0 and %exiting1 is replace by codeReplacer)
  store i32 %a, i32* %arg
  store i32 %B, i32* %arg
  br label %after

exit1:
  br label %after

exit2:
  %C = phi i32 [ %c, %exiting1 ]
  store i32 %c, i32* %arg
  store i32 %C, i32* %arg
  br label %return

after:
  %D = phi i32 [ %b, %exit0 ], [ %c, %exit1 ]
  store i32 %a, i32* %arg
  store i32 %D, i32* %arg
  br label %return

return:
  ret void
}
