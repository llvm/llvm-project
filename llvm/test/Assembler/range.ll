; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define i8 @neg1_zero(ptr %x) {
; CHECK-LABEL: define i8 @neg1_zero
; CHECK-SAME: (ptr [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[Y:%.*]] = load i8, ptr [[X]], align 1, !range [[RNG0:![0-9]+]]
; CHECK-NEXT:    ret i8 [[Y]]
;
entry:
  %y = load i8, ptr %x, align 1, !range !0
  ret i8 %y
}

define <2 x i8> @neg1_zero_vector(ptr %x) {
; CHECK-LABEL: define <2 x i8> @neg1_zero_vector
; CHECK-SAME: (ptr [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[Y:%.*]] = load <2 x i8>, ptr [[X]], align 1, !range [[RNG0]]
; CHECK-NEXT:    ret <2 x i8> [[Y]]
;
entry:
  %y = load <2 x i8>, ptr %x, align 1, !range !0
  ret <2 x i8> %y
}

!0 = !{i8 -1, i8 0}

;.
; CHECK: [[RNG0]] = !{i8 -1, i8 0}
;.
