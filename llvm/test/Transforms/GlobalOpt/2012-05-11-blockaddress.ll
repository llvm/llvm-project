; RUN: opt < %s -globalopt -S | FileCheck %s
; Check that the mere presence of a blockaddress doesn't prevent -globalopt
; from promoting @f to fastcc.

; CHECK-LABEL: define{{.*}}fastcc{{.*}}@f(
define internal i8* @f() {
  ret i8* blockaddress(@f, %L1)
L1:
  ret i8* null
}

define void @g() {
  ; CHECK: call{{.*}}fastcc{{.*}}@f
  %p = call i8* @f()
  ret void
}
