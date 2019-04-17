; RUN: opt -S -rewrite-statepoints-for-gc < %s | FileCheck %s
; RUN: opt -S -passes=rewrite-statepoints-for-gc < %s | FileCheck %s

declare void @f()
declare i32 @personality_function()

define void @test_id() gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: @test_id(
entry:
; CHECK-LABEL: entry:
; CHECK: invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 100, i32 0, void ()* @f
  invoke void @f()  "statepoint-id"="100" to label %normal_return unwind label %exceptional_return

normal_return:
  ret void

exceptional_return:
  %landing_pad4 = landingpad {i8*, i32} cleanup
  ret void
}

define void @test_num_patch_bytes() gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: @test_num_patch_bytes(
entry:
; CHECK-LABEL: entry:
; CHECK: invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 99, void ()* @f,
  invoke void @f()  "statepoint-num-patch-bytes"="99" to label %normal_return unwind label %exceptional_return

normal_return:
  ret void

exceptional_return:
  %landing_pad4 = landingpad {i8*, i32} cleanup
  ret void
}

declare void @do_safepoint()
define void @gc.safepoint_poll() {
entry:
  call void @do_safepoint()
  ret void
}

; CHECK-NOT: statepoint-id
; CHECK-NOT: statepoint-num-patch_bytes
