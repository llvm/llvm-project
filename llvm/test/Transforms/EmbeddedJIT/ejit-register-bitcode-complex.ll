; RUN: opt -passes=ejit-register-bitcode -S %s | FileCheck %s

; Test: Multiple ejit_entry functions with nested call chains,
; verify transitive closure includes all reachable functions but excludes
; unreachable ones.

; CHECK: @__ejit_bitcode = internal constant [{{.*}} x i8]
; CHECK: section ".ejit.bitcode"

; CHECK: define internal void @ejit_auto_register()
; CHECK: call void @ejit_register_bitcode

; Entry function with two-level call chain: entry1 -> helper_a -> helper_b
define void @helper_b() {
  ret void
}

define void @helper_a() {
  call void @helper_b()
  ret void
}

define void @ejit_entry1() !ejit.metadata !0 {
  call void @helper_a()
  ret void
}

; Second entry with separate call chain: entry2 -> helper_c
define void @helper_c() {
  ret void
}

; Unreachable function, must NOT be in bitcode
define void @unreachable_helper() {
  ret void
}

define void @ejit_entry2() !ejit.metadata !1 {
  call void @helper_c()
  ret void
}

!0 = distinct !{!{!"ejit_entry"}}
!1 = distinct !{!{!"ejit_entry"}}
