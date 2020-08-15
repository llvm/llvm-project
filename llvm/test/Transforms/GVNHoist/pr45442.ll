; RUN: opt < %s -gvn-hoist -S | FileCheck %s

; gvn-hoist shouldn't crash in this case.
; CHECK-LABEL: @func(i1 %b)
; CHECK:       entry:
; CHECK-NEXT:  br i1
; CHECK:  bb1:
; CHECK-NEXT:  ret void
; CHECK:  bb2:
; CHECK-NEXT:  call
; CHECK-NEXT:  call
; CHECK-NEXT:  ret void

define void @v_1_0() #0 {
entry:
  ret void
}

define void @func(i1 %b) {
entry:
  br i1 %b, label %bb1, label %bb2

bb1:
  ret void

bb2:
  call void @v_1_0()
  call void @v_1_0()
  ret void
}

attributes #0 = { nounwind readonly }
