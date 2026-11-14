; NOTE: Do not autogenerate. Check metadata ownership and removal of empty split blocks.
; RUN: opt -S -passes='require<domtree>,break-crit-edges,verify<domtree>' -verify-each %s | FileCheck %s
; RUN: opt -S -passes='require<domtree>,break-crit-edges,simplifycfg' -verify-each %s | FileCheck %s --check-prefix=FOLD
; RUN: opt -S -passes='require<domtree>,require<loops>,break-crit-edges,verify<domtree>,verify<loops>' -verify-each %s | FileCheck %s

; A split self-backedge becomes the latch. The old terminator and split exits
; must not retain that loop's metadata. Exercise both successor orders.

; The stale attachment on the old branch would keep the empty split blocks.
; FOLD-LABEL: define i32 @backedge_first(
; FOLD: loop:
; FOLD: br i1 %again, label %loop, label %exit, !llvm.loop !{{[0-9]+}}
; FOLD-EMPTY:
; FOLD-NEXT: exit:
; FOLD: ret i32

define i32 @backedge_first(i1 %enter, i1 %again, i32 %seed) {
; CHECK-LABEL: define i32 @backedge_first(
; CHECK: loop:
; CHECK: br i1 %again, label %[[BACK:[^, ]+]], label %[[EXIT:[^, ]+]]{{$}}
; CHECK: [[EXIT]]:
; CHECK-NEXT: br label %exit{{$}}
; CHECK: [[BACK]]:
; CHECK-NEXT: br label %loop, !llvm.loop ![[ID:[0-9]+]]
entry:
  br i1 %enter, label %loop, label %exit
loop:
  %x = phi i32 [ %seed, %entry ], [ %next, %loop ]
  %next = add i32 %x, 1
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %result = phi i32 [ %seed, %entry ], [ %next, %loop ]
  ret i32 %result
}

define i32 @exit_first(i1 %enter, i1 %done, i32 %seed) {
; CHECK-LABEL: define i32 @exit_first(
; CHECK: loop:
; CHECK: br i1 %done, label %[[EXIT:[^, ]+]], label %[[BACK:[^, ]+]]{{$}}
; CHECK: [[BACK]]:
; CHECK-NEXT: br label %loop, !llvm.loop ![[ID]]
; CHECK: [[EXIT]]:
; CHECK-NEXT: br label %exit{{$}}
entry:
  br i1 %enter, label %loop, label %exit
loop:
  %x = phi i32 [ %seed, %entry ], [ %next, %loop ]
  %next = add i32 %x, 1
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %result = phi i32 [ %seed, %entry ], [ %next, %loop ]
  ret i32 %result
}

define i32 @two_backedges(i32 %seed) {
; CHECK-LABEL: define i32 @two_backedges(
; CHECK: loop:
; CHECK: switch i32 %next, label %exit [
; CHECK-NEXT: i32 1, label %[[BACK1:[^ ]+]]
; CHECK-NEXT: i32 2, label %[[BACK2:[^ ]+]]
; CHECK-NEXT: ]{{$}}
; CHECK: [[BACK2]]:
; CHECK-NEXT: br label %loop, !llvm.loop ![[ID]]
; CHECK: [[BACK1]]:
; CHECK-NEXT: br label %loop, !llvm.loop ![[ID]]
entry:
  br label %loop
loop:
  %x = phi i32 [ %seed, %entry ], [ %next, %loop ], [ %next, %loop ]
  %next = add i32 %x, 1
  switch i32 %next, label %exit [ i32 1, label %loop
                                i32 2, label %loop ], !llvm.loop !0
exit:
  ret i32 %next
}

; Metadata also moves to the new latch of an ordinary multi-block loop.
define void @ordinary_loop(i1 %again, i1 %enter) {
; CHECK-LABEL: define void @ordinary_loop(
; CHECK: latch:
; CHECK-NEXT: br i1 %again, label %[[BACK:[^, ]+]], label %[[EXIT:[^, ]+]]{{$}}
; CHECK: [[EXIT]]:
; CHECK-NEXT: br label %exit{{$}}
; CHECK: [[BACK]]:
; CHECK-NEXT: br label %header, !llvm.loop ![[OUTER:[0-9]+]]
entry:
  br i1 %enter, label %header, label %exit
header:
  br label %latch
latch:
  br i1 %again, label %header, label %exit, !llvm.loop !1
exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!"llvm.loop.unroll.count", i32 2}

; The old terminator is the latch of both the inner and outer loops.
define void @shared_latch(i32 %choice) {
; CHECK-LABEL: define void @shared_latch(
; CHECK: inner:
; CHECK: switch i32 %choice, label %exit [
; CHECK-NEXT: i32 0, label %[[INNER:[^ ]+]]
; CHECK-NEXT: i32 1, label %[[OUTER:[^ ]+]]
; CHECK-NEXT: ]{{$}}
; CHECK: [[OUTER]]:
; CHECK-NEXT: br label %outer, !llvm.loop ![[ID]]
; CHECK: [[INNER]]:
; CHECK-NEXT: br label %inner, !llvm.loop ![[ID]]
entry:
  br label %outer
outer:
  br label %inner
inner:
  switch i32 %choice, label %exit [ i32 0, label %inner
                                  i32 1, label %outer ], !llvm.loop !0
exit:
  ret void
}

; Exercise the other order of the shared latch's backedges.
define void @shared_latch_outer_first(i32 %choice) {
; CHECK-LABEL: define void @shared_latch_outer_first(
; CHECK: inner:
; CHECK: switch i32 %choice, label %exit [
; CHECK-NEXT: i32 0, label %[[OUTER:[^ ]+]]
; CHECK-NEXT: i32 1, label %[[INNER:[^ ]+]]
; CHECK-NEXT: ]{{$}}
; CHECK: [[INNER]]:
; CHECK-NEXT: br label %inner, !llvm.loop ![[ID]]
; CHECK: [[OUTER]]:
; CHECK-NEXT: br label %outer, !llvm.loop ![[ID]]
entry:
  br label %outer
outer:
  br label %inner
inner:
  switch i32 %choice, label %exit [ i32 0, label %outer
                                  i32 1, label %inner ], !llvm.loop !0
exit:
  ret void
}
; Without a reachable loop, retain the existing conservative copying behavior.
define void @unreachable_loop(i1 %again) {
; CHECK-LABEL: define void @unreachable_loop(
; CHECK: loop:
; CHECK-NEXT: br i1 %again, label %[[BACK:[^, ]+]], label %[[EXIT:[^, ]+]], !llvm.loop ![[ID]]
; CHECK: [[EXIT]]:
; CHECK-NEXT: br label %exit, !llvm.loop ![[ID]]
; CHECK: [[BACK]]:
; CHECK-NEXT: br label %loop, !llvm.loop ![[ID]]
entry:
  br label %exit
unreachable_entry:
  br label %loop
loop:
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
