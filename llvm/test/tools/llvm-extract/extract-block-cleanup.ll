; RUN: llvm-extract -S -bb "foo:region_start;extractonly;cleanup;fallback;region_end" --replace-with-call %s | FileCheck %s


; CHECK-LABEL: define void @foo(ptr %arg, i1 %c) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %codeRepl, label %outsideonly
; CHECK-EMPTY:
; CHECK-NEXT:  outsideonly:
; CHECK-NEXT:    store i32 0, ptr %arg, align 4
; CHECK-NEXT:    br label %cleanup
; CHECK-EMPTY:
; CHECK-NEXT:  codeRepl:
; CHECK-NEXT:    %targetBlock = call i1 @foo.region_start(ptr %arg)
; CHECK-NEXT:    br i1 %targetBlock, label %cleanup.return_crit_edge, label %region_end.split
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    br label %extractonly
; CHECK-EMPTY:
; CHECK-NEXT:  extractonly:
; CHECK-NEXT:    store i32 1, ptr %arg, align 4
; CHECK-NEXT:    br label %cleanup
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup:
; CHECK-NEXT:    %dest = phi i8 [ 0, %outsideonly ], [ 1, %extractonly ]
; CHECK-NEXT:    switch i8 %dest, label %fallback [
; CHECK-NEXT:      i8 0, label %cleanup.return_crit_edge
; CHECK-NEXT:      i8 1, label %region_end
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup.return_crit_edge:
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  fallback:
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  region_end:
; CHECK-NEXT:    br label %region_end.split
; CHECK-EMPTY:
; CHECK-NEXT:  region_end.split:
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  outsidecont:
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  return:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }


; CHECK-LABEL: define internal i1 @foo.region_start(ptr %arg) {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %region_start
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    br label %extractonly
; CHECK-EMPTY:
; CHECK-NEXT:  extractonly:
; CHECK-NEXT:    store i32 1, ptr %arg, align 4
; CHECK-NEXT:    br label %cleanup
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup:
; CHECK-NEXT:    %dest = phi i8 [ 1, %extractonly ]
; CHECK-NEXT:    switch i8 %dest, label %fallback [
; CHECK-NEXT:      i8 0, label %cleanup.return_crit_edge.exitStub
; CHECK-NEXT:      i8 1, label %region_end
; CHECK-NEXT:    ]
; CHECK-EMPTY:
; CHECK-NEXT:  fallback:
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  region_end:
; CHECK-NEXT:    br label %region_end.split.exitStub
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup.return_crit_edge.exitStub:
; CHECK-NEXT:    ret i1 true
; CHECK-EMPTY:
; CHECK-NEXT:  region_end.split.exitStub:
; CHECK-NEXT:    ret i1 false
; CHECK-NEXT:  }



define void @foo(ptr %arg, i1 %c) {
entry:
  br i1 %c, label %region_start, label %outsideonly

outsideonly:
  store i32 0, ptr %arg, align 4
  br label %cleanup

region_start:
  br label %extractonly

extractonly:
  store i32 1, ptr %arg, align 4
  br label %cleanup

cleanup:
  %dest = phi i8 [0, %outsideonly], [1, %extractonly]
  switch i8 %dest, label %fallback [
    i8 0, label %return
    i8 1, label %region_end
  ]

fallback:
  unreachable

region_end:
  br label %return

outsidecont:
  br label %return

return:
  ret void
}
