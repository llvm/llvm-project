; RUN: opt -passes=ejit-register-bitcode -S %s | FileCheck %s

; CHECK: @__ejit_bitcode = internal constant [{{.*}} x i8]
; CHECK: section ".ejit.bitcode"

; CHECK: define internal void @ejit_auto_register()
; CHECK: call void @ejit_register_bitcode

define void @my_func1() {
  ret void
}

define void @my_func2() {
  ret void
}

define void @my_ejit_entry() !ejit.metadata !0 {
  call void @my_func1()
  call void @my_func2()
  ret void
}

!0 = distinct !{!1}
!1 = !{!"ejit_entry"}
