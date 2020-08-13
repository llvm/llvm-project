; RUN: opt < %s -gvn-hoist -S | FileCheck %s

; gvn-hoist shouldn't crash in this case.
; CHECK-LABEL: @func()
; CHECK:       entry:
; CHECK-NEXT:  br i1
; CHECK:  bb1:
; CHECK-NEXT:  unreachable
; CHECK:  bb2:
; CHECK-NEXT:  call
; CHECK-NEXT:  call
; CHECK-NEXT:  unreachable

define void @v_1_0() #0 {
entry:
  ret void
}

define void @func()  {
entry:
  br i1 undef, label %bb1, label %bb2

bb1:
  unreachable

bb2:
  call void @v_1_0()
  call void @v_1_0()
  unreachable
}

attributes #0 = { nounwind readonly }
