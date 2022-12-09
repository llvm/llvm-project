; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

%0 = type opaque
declare void @g()
declare %0* @foo0()
declare i8 @foo1()
declare void @noreturn_func()

; Operand bundles uses are like regular uses, and need to be dominated
; by their defs.

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

define void @f_gc_transition(i32* %ptr) {
; CHECK: Multiple gc-transition operand bundles
; CHECK-NEXT: call void @g() [ "gc-transition"(i32 42, i64 100, i32 %x), "gc-transition"(float 0.000000e+00, i64 100, i32 %l) ]
; CHECK-NOT: call void @g() [ "gc-transition"(i32 42, i64 120, i32 %x) ]

 entry:
  %l = load i32, i32* %ptr
  call void @g() [ "gc-transition"(i32 42, i64 100, i32 %x), "gc-transition"(float 0.0, i64 100, i32 %l) ]
  call void @g() [ "gc-transition"(i32 42, i64 120) ]  ;; The verifier should not complain about this one
  %x = add i32 42, 1
  ret void
}

define void @f_clang_arc_attachedcall() {
; CHECK: requires one function as an argument
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"() ]
; CHECK-NEXT: Multiple "clang.arc.attachedcall" operand bundles
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: must call a function returning a pointer
; CHECK-NEXT: call i8 @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: or a non-returning function
; CHECK-NEXT: call void @g() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
; CHECK-NEXT: requires one function as an argument
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* null) ]
; CHECK-NEXT: requires one function as an argument
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"(i64 0) ]
; CHECK-NEXT: invalid function argument
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"(i8 ()* @foo1) ]
; CHECK-NEXT: invalid function argument
; CHECK-NEXT: call %0* @foo0() [ "clang.arc.attachedcall"(void (i1)* @llvm.assume) ]

  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_unsafeClaimAutoreleasedReturnValue) ]
  call %0* @foo0() [ "clang.arc.attachedcall"() ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call i8 @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void @noreturn_func() #0 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void @g() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8* (i8*)* null) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i64 0) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(i8 ()* @foo1) ]
  call %0* @foo0() [ "clang.arc.attachedcall"(void (i1)* @llvm.assume) ]
  ret void
}

declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_unsafeClaimAutoreleasedReturnValue(i8*)
declare void @llvm.assume(i1)

attributes #0 = { noreturn }
