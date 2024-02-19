; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @test_kcfi_bundle(i64 %arg0, i32 %arg1, ptr %arg2) {
; CHECK: Multiple kcfi operand bundles
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42), "kcfi"(i32 42) ]
  call void %arg2() [ "kcfi"(i32 42), "kcfi"(i32 42) ]

; CHECK: Kcfi bundle operand must be an i32 constant
; CHECK-NEXT: call void %arg2() [ "kcfi"(i64 42) ]
  call void %arg2() [ "kcfi"(i64 42) ]

; CHECK-NOT: call void
  call void %arg2() [ "kcfi"(i32 42) ] ; OK
  call void %arg2() [ "kcfi"(i32 42) ] ; OK
  ret void
}
