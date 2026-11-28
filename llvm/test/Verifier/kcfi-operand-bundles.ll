; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

@data = private global i8 0

define void @test_kcfi_bundle(i64 %arg0, i32 %arg1, ptr %arg2) {
; CHECK: Multiple kcfi operand bundles
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42), "kcfi"(i32 42) ]
  call void %arg2() [ "kcfi"(i32 42), "kcfi"(i32 42) ]

; CHECK: Kcfi bundle operand must be an i32 constant
; CHECK-NEXT: call void %arg2() [ "kcfi"(i64 42) ]
  call void %arg2() [ "kcfi"(i64 42) ]

; CHECK: Expected one or three kcfi bundle operands
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr null) ]
  call void %arg2() [ "kcfi"(i32 42, ptr null) ]

; CHECK: Kcfi bundle data operand must be an unqualified pointer
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, i32 0, i32 0) ]
  call void %arg2() [ "kcfi"(i32 42, i32 0, i32 0) ]

; CHECK: Kcfi bundle data operand must be an unqualified pointer
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr addrspace(1) null, i32 2) ]
  call void %arg2() [ "kcfi"(i32 42, ptr addrspace(1) null, i32 2) ]

; CHECK: Kcfi bundle flags operand must be an i32 constant
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr null, i32 %arg1) ]
  call void %arg2() [ "kcfi"(i32 42, ptr null, i32 %arg1) ]

; CHECK: Unknown kcfi bundle flags
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr @data, i32 16) ]
  call void %arg2() [ "kcfi"(i32 42, ptr @data, i32 16) ]

; CHECK: Kcfi bundle data operand cannot be null without minimal runtime flag
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr null, i32 0) ]
  call void %arg2() [ "kcfi"(i32 42, ptr null, i32 0) ]

; CHECK: Kcfi bundle preserve-all flag requires recoverable minimal runtime
; CHECK-NEXT: call void %arg2() [ "kcfi"(i32 42, ptr null, i32 6) ]
  call void %arg2() [ "kcfi"(i32 42, ptr null, i32 6) ]

; CHECK-NOT: call void
  call void %arg2() [ "kcfi"(i32 42) ] ; OK
  call void %arg2() [ "kcfi"(i32 42) ] ; OK
  call void %arg2() [ "kcfi"(i32 42, ptr @data, i32 0) ] ; OK
  call void %arg2() [ "kcfi"(i32 42, ptr null, i32 2) ] ; OK
  call void %arg2() [ "kcfi"(i32 42, ptr null, i32 7) ] ; OK
  ret void
}
