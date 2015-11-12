; RUN: not opt -verify < %s 2>&1 | FileCheck %s

; Operand bundles uses are like regular uses, and need to be dominated
; by their defs.

declare void @g()

define void @f0(i32* %ptr) {
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %x = add i32 42, 1
; CHECK-NEXT:  call void @g() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float 0.000000e+00, i64 100, i32 %l) ]

 entry:
  %l = load i32, i32* %ptr
  call void @g() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float 0.0, i64 100, i32 %l) ]
  %x = add i32 42, 1
  ret void
}

define void @f1(i32* %ptr) personality i8 3 {
; CHECK: Instruction does not dominate all uses!
; CHECK-NEXT:  %x = add i32 42, 1
; CHECK-NEXT:  invoke void @g() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float 0.000000e+00, i64 100, i32 %l) ]

 entry:
  %l = load i32, i32* %ptr
  invoke void @g() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float 0.0, i64 100, i32 %l) ] to label %normal unwind label %exception

exception:
  %cleanup = landingpad i8 cleanup
  br label %normal

normal:
  %x = add i32 42, 1
  ret void
}

define void @f_deopt(i32* %ptr) {
; CHECK: Multiple deopt operand bundles
; CHECK-NEXT: call void @g() [ "deopt"(i32 42, i64 100, i32 %x), "deopt"(float 0.000000e+00, i64 100, i32 %l) ]
; CHECK-NOT: call void @g() [ "deopt"(i32 42, i64 120, i32 %x) ]

 entry:
  %l = load i32, i32* %ptr
  call void @g() [ "deopt"(i32 42, i64 100, i32 %x), "deopt"(float 0.0, i64 100, i32 %l) ]
  call void @g() [ "deopt"(i32 42, i64 120) ]  ;; The verifier should not complain about this one
  %x = add i32 42, 1
  ret void
}
