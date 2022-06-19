; RUN: opt -passes=inline < %s -S | FileCheck %s

; CHECK: define {{.*}}@caller
; CHECK: define {{.*}}@f1
; CHECK-NOT: define {{.*}}@f2
; CHECK-NOT: define {{.*}}@f3
; CHECK-NOT: define {{.*}}@f4
; CHECK-NOT: define {{.*}}@f5
; CHECK: define {{.*}}@f6
; CHECK-NOT: define {{.*}}@f7
; CHECK-NOT: define {{.*}}@f8

$c1 = comdat any
$c2 = comdat any
$c3 = comdat any

define void @caller() {
  call void @f1()
  call void @f2()
  call void @f3()
  call void @f4()
  call void @f5()
  call void @f6()
  call void @f7()
  call void @f8()
  ret void
}

define void @f1() {
  ret void
}

define internal void @f2() {
  ret void
}

define private void @f3() {
  ret void
}

define linkonce_odr void @f4() {
  ret void
}

define linkonce_odr void @f5() comdat($c1) {
  ret void
}

define linkonce_odr void @f6() comdat($c2) {
  ret void
}

define linkonce_odr void @g() comdat($c2) {
  ret void
}

define linkonce_odr void @f7() comdat($c3) {
  ret void
}

define linkonce_odr void @f8() comdat($c3) {
  ret void
}
