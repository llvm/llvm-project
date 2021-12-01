; RUN: llvm-extract -S -bb "foo:region_start;extractonly;both;fallback;region_end" %s --bb-keep-functions --bb-keep-blocks | FileCheck %s


; CHECK-LABEL: define void @foo(
;
; CHECK:       outsideonly:
; CHECK-NEXT:    store i32 0, i32* %arg, align 4
; CHECK-NEXT:    br label %both
;
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @foo.region_start(i32* %arg)
; CHECK-NEXT:    br label %return
;
; CHECK:       extractonly:
; CHECK-NEXT:    store i32 1, i32* %arg, align 4
; CHECK-NEXT:    br label %both
;
; CHECK:       both:
; CHECK-NEXT:    %dest = phi i8 [ 0, %outsideonly ], [ 1, %extractonly ]
; CHECK-NEXT:    switch


; CHECK-LABEL: define internal void @foo.region_start(i32* %arg) {
; CHECK:         br label %region_start
;
; CHECK:       return.exitStub:
; CHECK-NEXT:    ret void
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    br label %extractonly
; CHECK-EMPTY:
; CHECK-NEXT:  extractonly:
; CHECK-NEXT:    store i32 1, i32* %arg, align 4
; CHECK-NEXT:    br label %both
; CHECK-EMPTY:
; CHECK-NEXT:  both:
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
; CHECK-NEXT:  }


define void @foo(i32* %arg, i1 %c) {
entry:
  br i1 %c, label %region_start, label %outsideonly

outsideonly:
  store i32 0, i32* %arg, align 4
  br label %both

region_start:
  ;store i32 %arg, i32* %tmp1, align 4
  ;%tmp2 = load i32, i32* %tmp1, align 4
  ;%tmp3 = icmp sgt i32 %tmp2, 0
  ;store i8 1, i8* %dest, align 4
  ;br i1 %tmp3, label %extractonly, label %bb7
  br label %extractonly

extractonly:
  store i32 1, i32* %arg, align 4
  br label %both

both:
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
