; RUN: opt -disable-output -print-mustexecute < %s 2>&1 | FileCheck %s

@c = global i16 0, align 2

; CHECK-LABEL: define void @latch_cycle_irreducible
; CHECK: store i16 5, ptr @c, align 2{{$}}
define void @latch_cycle_irreducible() {
entry:
  br label %loop

loop:                                             ; preds = %loop.latch, %entry
  %v = phi i32 [ 10, %entry ], [ 0, %loop.latch ]
  %c = icmp eq i32 %v, 0
  br i1 %c, label %loop.exit, label %loop.cont

loop.cont:                                        ; preds = %loop
  br i1 false, label %loop.irreducible, label %loop.latch

loop.irreducible:                                 ; preds = %loop.latch, %loop.cont
  store i16 5, ptr @c, align 2
  br label %loop.latch

loop.latch:                                       ; preds = %loop.irreducible, %loop.cont
  br i1 false, label %loop.irreducible, label %loop

loop.exit:                                        ; preds = %loop
  ret void
}

; CHECK-LABEL: define void @latch_cycle_reducible
; CHECK: store i16 5, ptr @c, align 2{{$}}
define void @latch_cycle_reducible() {
entry:
  br label %loop

loop:                                             ; preds = %loop.latch, %entry
  %v = phi i32 [ 10, %entry ], [ 0, %loop.latch ]
  %c = icmp eq i32 %v, 0
  br i1 %c, label %loop.exit, label %loop2

loop2:                                            ; preds = %loop.latch, %loop
  br i1 false, label %loop2.cont, label %loop.latch

loop2.cont:                                       ; preds = %loop2
  store i16 5, ptr @c, align 2
  br label %loop.latch

loop.latch:                                       ; preds = %loop2.cont, %loop2
  br i1 false, label %loop2, label %loop

loop.exit:                                        ; preds = %loop
  ret void
}
