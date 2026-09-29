; RUN: opt < %s -passes=reassociate -stats -S 2>&1 | FileCheck %s
; REQUIRES: asserts

define i8 @longchain(i8 %in1, i8 %in2, i8 %in3, i8 %in4, i8 %in5, i8 %in6, i8 %in7, i8 %in8, i8 %in9, i8 %in10, i8 %in11, i8 %in12, i8 %in13, i8 %in14, i8 %in15, i8 %in16, i8 %in17, i8 %in18, i8 %in19, i8 %in20) {
  %tmp1 = add i8 %in1, %in2
  %tmp2 = add i8 %tmp1, %in3
  %tmp3 = add i8 %tmp2, %in4
  %tmp4 = add i8 %tmp3, %in3
  %tmp5 = add i8 %tmp4, %in4
  %tmp6 = add i8 %tmp5, %in5
  %tmp7 = add i8 %tmp6, %in6
  %tmp8 = add i8 %tmp7, %in7
  %tmp9 = add i8 %tmp8, %in8
  %tmp10 = add i8 %tmp9, %in9
  %tmp11 = add i8 %tmp10, %in10
  %tmp12 = add i8 %tmp11, %in11
  %tmp13 = add i8 %tmp12, %in12
  %tmp14 = add i8 %tmp13, %in13
  %tmp15 = add i8 %tmp14, %in14
  %tmp16 = add i8 %tmp15, %in15
  %tmp17 = add i8 %tmp16, %in16
  %tmp18 = add i8 %tmp17, %in17
  %tmp19 = add i8 %tmp18, %in18
  %tmp20 = add i8 %tmp19, %in19
  %tmp21 = add i8 %tmp20, %in20
  ret i8 %tmp20
}

; Check the number of instructions reassociated is in the tens not the hundreds.
; At the time of writing, the exact numbers were:
; Bad order: 220 reassociate - Number of insts reassociated
; Good order: 55 reassociate - Number of insts reassociated
;
; CHECK: {{^[1-9][0-9]}} reassociate - Number of insts reassociated

; Additionally check that we made at least three changes.
; CHECK:      {{^ *[3-9]}} reassociate - Number of multiplies factored
