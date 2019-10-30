; RUN: not opt -verify < %s 2>&1 | FileCheck %s

declare void @g()

define void @f_deopt(i64 %arg0, i32 %arg1) {
; CHECK: Multiple ptrauth operand bundles
; CHECK-NEXT: call void @g() [ "ptrauth"(i32 42, i64 100), "ptrauth"(i32 42, i64 %arg0) ]
; CHECK: Ptrauth bundle key operand must be an i32 constant
; CHECK-NEXT: call void @g() [ "ptrauth"(i32 %arg1, i64 120) ]
; CHECK: Ptrauth bundle key operand must be an i32 constant
; CHECK-NEXT: call void @g() [ "ptrauth"(i64 42, i64 120) ]
; CHECK: Ptrauth bundle discriminator operand must be an i64
; CHECK-NEXT: call void @g() [ "ptrauth"(i32 42, i32 120) ]
; CHECK-NOT: call void @g() [ "ptrauth"(i32 42, i64 120, i32 %x) ]

 entry:
  call void @g() [ "ptrauth"(i32 42, i64 100), "ptrauth"(i32 42, i64 %arg0) ]
  call void @g() [ "ptrauth"(i32 %arg1, i64 120) ]
  call void @g() [ "ptrauth"(i64 42, i64 120) ]
  call void @g() [ "ptrauth"(i32 42, i32 120) ]
  call void @g() [ "ptrauth"(i32 42, i64 120) ]  ;; The verifier should not complain about this one
  ret void
}
