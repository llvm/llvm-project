; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s

@g = common dso_local global i32 0, align 4
declare void @callee()
declare void @z(i32, i32, i32, i32)

define dso_local void @foo(ptr %p) #0 {
; CHECK-LABEL: foo:
; CHECK: bl OUTLINED_FUNCTION_1
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  call void @callee()
  store volatile i32 3, ptr @g, align 4
  ret void
}

define dso_local void @bar(ptr %p) #0 {
; CHECK-LABEL: bar:
; CHECK: bl OUTLINED_FUNCTION_1
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  call void @callee()
  store volatile i32 4, ptr @g, align 4
  ret void
}

; Leaf functions under "sign-return-address"="non-leaf" can be safely tail-call outlined without signing.
define dso_local void @leaf_tc1(ptr %p) #1 {
; CHECK-LABEL: leaf_tc1:
; CHECK: b OUTLINED_FUNCTION_0
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  store volatile i32 3, ptr %p, align 4
  store volatile i32 4, ptr %p, align 4
  ret void
}

define dso_local void @leaf_tc2(ptr %p) #1 {
; CHECK-LABEL: leaf_tc2:
; CHECK: b OUTLINED_FUNCTION_0
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  store volatile i32 3, ptr %p, align 4
  store volatile i32 4, ptr %p, align 4
  ret void
}

; Tail calls under "sign-return-address"="non-leaf" can also be outlined without signing.
define void @tc_call_a() #1 {
; CHECK-LABEL: tc_call_a:
; CHECK: b OUTLINED_FUNCTION_2
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}

define void @tc_call_b() #1 {
; CHECK-LABEL: tc_call_b:
; CHECK: b OUTLINED_FUNCTION_2
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Tail calls under "sign-return-address"="all" must not be outlined to avoid corrupting LR.
define dso_local void @all_tc1(ptr %p) #0 {
; CHECK-LABEL: all_tc1:
; CHECK-NOT: b OUTLINED_FUNCTION_
entry:
  store volatile i32 10, ptr %p, align 4
  store volatile i32 20, ptr %p, align 4
  store volatile i32 30, ptr %p, align 4
  store volatile i32 40, ptr %p, align 4
  ret void
}

define dso_local void @all_tc2(ptr %p) #0 {
; CHECK-LABEL: all_tc2:
; CHECK-NOT: b OUTLINED_FUNCTION_
entry:
  store volatile i32 10, ptr %p, align 4
  store volatile i32 20, ptr %p, align 4
  store volatile i32 30, ptr %p, align 4
  store volatile i32 40, ptr %p, align 4
  ret void
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT: hint #25
; CHECK-NOT: paciasp
; CHECK: ret

; CHECK-LABEL: OUTLINED_FUNCTION_1:
; CHECK: hint #25
; CHECK: hint #29
; CHECK: b callee

; CHECK-LABEL: OUTLINED_FUNCTION_2:
; CHECK-NOT: hint #25
; CHECK-NOT: paciasp
; CHECK: b z

attributes #0 = { "sign-return-address"="all" }
attributes #1 = { "sign-return-address"="non-leaf" }
