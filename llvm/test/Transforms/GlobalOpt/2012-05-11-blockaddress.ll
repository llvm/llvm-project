; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; Check that the mere presence of a blockaddress doesn't prevent -globalopt
; from promoting @f to fastcc.

; CHECK-LABEL: define{{.*}}fastcc{{.*}}@f(
define internal ptr @f() {
  ret ptr blockaddress(@f, %L1)
L1:
  ret ptr null
}

define void @g() {
  ; CHECK: call{{.*}}fastcc{{.*}}@f
  %p = call ptr @f()
  ret void
}
