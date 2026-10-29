; RUN: opt -S -passes='lower-switch,verify' %s | FileCheck %s

; With no default traffic, each comparison receives the sum of its cases.
; CHECK-LABEL: define void @zero_default(
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[ROOT:![0-9]+]]
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[RIGHT:![0-9]+]]
define void @zero_default(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 0, label %a
                             i32 1, label %b
                             i32 2, label %c], !prof !0
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
 c: call void @use(i32 2)
  br label %exit
exit: ret void
}

; The default values can reach either side of the tree. Their split is unknown.
; CHECK-LABEL: define void @unknown_default_split(
; CHECK-NOT: !prof
; CHECK: ret void
define void @unknown_default_split(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 0, label %a
                             i32 2, label %b], !prof !1
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
exit: ret void
}

; A single comparison can retain both weights, including the expected marker.
; CHECK-LABEL: define void @single_case(
; CHECK: br i1 %SwitchLeaf, label %a, label %exit, !prof [[SINGLE:![0-9]+]]
define void @single_case(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 7, label %a], !prof !2
 a: call void @use(i32 0)
  br label %exit
exit: ret void
}

; An i2 has only four values, so all default traffic must take value 1.
; CHECK-LABEL: define void @known_default_value(
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[KNOWN:![0-9]+]]
define void @known_default_value(i2 %v) {
entry:
  switch i2 %v, label %exit [i2 -2, label %a
                            i2 -1, label %b
                            i2 0, label %c], !prof !3
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
 c: call void @use(i32 2)
  br label %exit
exit: ret void
}

; Replacing an unreachable default with a popular destination retains all of
; that destination's case weights, including noncontiguous case values.
; CHECK-LABEL: define void @popular_default(
; CHECK: br i1 %SwitchLeaf, label %b, label %a, !prof [[POPULAR:![0-9]+]]
define void @popular_default(i32 %v) {
entry:
  switch i32 %v, label %dead [i32 0, label %a
                             i32 1, label %a
                             i32 2, label %b
                             i32 3, label %a], !prof !4
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
exit: ret void
dead: unreachable
}

; Explicit cases targeting the default still have individually known weights.
; CHECK-LABEL: define void @explicit_default_case(
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[POPULAR]]
define void @explicit_default_case(i2 %v) {
entry:
  switch i2 %v, label %exit [i2 -2, label %a
                            i2 -1, label %exit
                            i2 0, label %b], !prof !3
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
exit: ret void
}

; Adjacent values sharing a destination retain their combined mass.
; CHECK-LABEL: define void @clustered_cases(
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[POPULAR]]
define void @clustered_cases(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 0, label %a
                             i32 1, label %a
                             i32 3, label %b
                             i32 4, label %c], !prof !4
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
 c: call void @use(i32 2)
  br label %exit
exit: ret void
}

; Sums use 64 bits and are scaled together when they do not fit in i32.
; CHECK-LABEL: define void @large_weights(
; CHECK: br i1 %Pivot{{.*}}, label {{.*}}, label {{.*}}, !prof [[LARGE:![0-9]+]]
define void @large_weights(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 0, label %a
                             i32 1, label %b
                             i32 2, label %c], !prof !5
 a: call void @use(i32 0)
  br label %exit
 b: call void @use(i32 1)
  br label %exit
 c: call void @use(i32 2)
  br label %exit
exit: ret void
}

; CHECK-LABEL: define void @all_zero(
; CHECK-NOT: !prof
; CHECK: ret void
define void @all_zero(i32 %v) {
entry:
  switch i32 %v, label %exit [i32 7, label %a], !prof !6
 a: call void @use(i32 0)
  br label %exit
exit: ret void
}

declare void @use(i32)
!0 = !{!"branch_weights", i32 0, i32 10, i32 20, i32 30}
!1 = !{!"branch_weights", i32 100, i32 10, i32 20}
!2 = !{!"branch_weights", !"expected", i32 40, i32 60}
!3 = !{!"branch_weights", i32 40, i32 10, i32 20, i32 30}
!4 = !{!"branch_weights", i32 0, i32 10, i32 20, i32 30, i32 40}
!5 = !{!"branch_weights", i32 0, i32 -1, i32 -1, i32 -1}
!6 = !{!"branch_weights", i32 0, i32 0}

; CHECK-DAG: [[ROOT]] = !{!"branch_weights", i32 10, i32 50}
; CHECK-DAG: [[RIGHT]] = !{!"branch_weights", i32 20, i32 30}
; CHECK-DAG: [[SINGLE]] = !{!"branch_weights", !"expected", i32 60, i32 40}
; CHECK-DAG: [[KNOWN]] = !{!"branch_weights", i32 10, i32 90}
; CHECK-DAG: [[POPULAR]] = !{!"branch_weights", i32 30, i32 70}
; CHECK-DAG: [[LARGE]] = !{!"branch_weights", i32 2147483647, i32 -1}
