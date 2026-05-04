; RUN: opt -passes=ejit-aot-module -S %s 2>&1 | FileCheck %s

; Verify diagnostic warning when ejit_entry references undeclared period array
; CHECK: EJit warning: function 'bad_entry' references ejit_period_arr 'cell' but it is not declared via ejit_period_arr_ind

@real_data = global [8 x i32] zeroinitializer, !ejit.metadata !10

define void @bad_entry(i32 %idx) !ejit.metadata !20 {
entry:
  %ptr = getelementptr i32, ptr @real_data, i32 %idx
  %val = load i32, ptr %ptr
  ret void
}

!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 8}}
!20 = distinct !{!{!"ejit_entry"}}
