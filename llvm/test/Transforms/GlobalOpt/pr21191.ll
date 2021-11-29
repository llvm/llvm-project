; RUN: opt < %s -passes=globalopt -S | FileCheck %s

$c = comdat any
; CHECK: $c = comdat any

define linkonce_odr void @foo() comdat($c) {
  ret void
}
; CHECK: define linkonce_odr void @foo() local_unnamed_addr comdat($c)

define linkonce_odr void @bar() comdat($c) {
  ret void
}
; CHECK: define linkonce_odr void @bar() local_unnamed_addr comdat($c)

define void @zed()  {
  call void @foo()
  ret void
}
