; RUN: opt < %s -passes=gvn-hoist -S | FileCheck %s

@g = external constant ptr

declare i32 @gxx_personality(...)
declare void @f0()
declare void @f1()
declare void @f2()

; Make sure opt won't crash and that the load
; is not hoisted from label6 to label4

;CHECK-LABEL: @func

define void @func() personality ptr @gxx_personality {
  invoke void @f0()
          to label %3 unwind label %1

1:
  %2 = landingpad { ptr, i32 }
          catch ptr @g
          catch ptr null
  br label %16

3:
  br i1 undef, label %4, label %10

;CHECK:       4:
;CHECK-NEXT:    %5 = load ptr, ptr undef, align 8
;CHECK-NEXT:    invoke void @f1()

4:
  %5 = load ptr, ptr undef, align 8
  invoke void @f1()
          to label %6 unwind label %1

;CHECK:       6:
;CHECK-NEXT:    %7 = load ptr, ptr undef, align 8
;CHECK-NEXT:    %8 = load ptr, ptr undef, align 8

6:
  %7 = load ptr, ptr undef, align 8
  %8 = load ptr, ptr undef, align 8
  br i1 true, label %9, label %17

9:
  invoke void @f0()
          to label %10 unwind label %1

10:
  invoke void @f2()
          to label %11 unwind label %1

11:
  %12 = invoke signext i32 undef(ptr null, i32 signext undef, i1 zeroext undef)
          to label %13 unwind label %14

13:
  unreachable

14:
  %15 = landingpad { ptr, i32 }
          catch ptr @g
          catch ptr null
  br label %16

16:
  unreachable

17:
  ret void

; uselistorder directives
  uselistorder ptr @f0, { 1, 0 }
  uselistorder label %1, { 0, 3, 1, 2 }
}
