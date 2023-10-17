; RUN: llc -mtriple=thumbv6-apple-darwin < %s | FileCheck %s
; r8869722

%struct.state = type { i32, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, ptr }
%struct.info = type { i32, i32, i32, i32, i32, i32, i32, ptr }

define void @t1(ptr %v) {
  %tmp6 = load i32, ptr null
  %tmp8 = alloca float, i32 %tmp6
  store i32 1, ptr null
  br label %return

return:                                           ; preds = %0
; CHECK: subs [[SCRATCH:r[0-7]]], r7, #7
; CHECK: subs [[SCRATCH]], #1
; CHECK: mov sp, [[SCRATCH]]
; CHECK-NEXT: pop
; CHECK-SAME: [[SCRATCH]]
  ret void
}
