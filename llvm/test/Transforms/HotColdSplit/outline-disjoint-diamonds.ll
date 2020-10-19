; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=-1 < %s 2>&1 | FileCheck %s

; CHECK-LABEL: define {{.*}}@fun
; CHECK: call {{.*}}@fun.cold.2(
; CHECK-NEXT: ret void
; CHECK: call {{.*}}@fun.cold.1(
; CHECK-NEXT: ret void
define void @fun() {
entry:
  br i1 undef, label %A.then, label %A.else

A.else:
  br label %A.then4

A.then4:
  br i1 undef, label %A.then5, label %A.end

A.then5:
  br label %A.cleanup

A.end:
  br label %A.cleanup

A.cleanup:
  %A.cleanup.dest.slot.0 = phi i32 [ 1, %A.then5 ], [ 0, %A.end ]
  unreachable

A.then:
  br i1 undef, label %B.then, label %B.else

B.then:
  ret void

B.else:
  br label %B.then4

B.then4:
  br i1 undef, label %B.then5, label %B.end

B.then5:
  br label %B.cleanup

B.end:
  br label %B.cleanup

B.cleanup:
  %B.cleanup.dest.slot.0 = phi i32 [ 1, %B.then5 ], [ 0, %B.end ]
  unreachable
}

; CHECK-LABEL: define {{.*}}@fun.cold.1(
; CHECK: %B.cleanup.dest.slot.0 = phi i32 [ 1, %B.then5 ], [ 0, %B.end ]
; CHECK-NEXT: unreachable

; CHECK-LABEL: define {{.*}}@fun.cold.2(
; CHECK: %A.cleanup.dest.slot.0 = phi i32 [ 1, %A.then5 ], [ 0, %A.end ]
; CHECK-NEXT: unreachable
