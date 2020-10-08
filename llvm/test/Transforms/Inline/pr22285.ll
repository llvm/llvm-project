; RUN: opt < %s -inline -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline),globaldce' -S | FileCheck %s

$f1 = comdat any
; CHECK-NOT: $f1 = comdat any

define void @f2() {
  call void @f1()
  ret void
}
; CHECK-LABEL: define void @f2

define linkonce_odr void @f1() comdat {
  ret void
}
; CHECK-NOT: define linkonce_odr void @f1() comdat
