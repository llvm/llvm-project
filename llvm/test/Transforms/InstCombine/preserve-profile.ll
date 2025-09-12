; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; CHECK: define i32 @LHSBin(i1 %C) !prof ![[PROF0:[0-9]]]
; CHECK: %V = select i1 %C, i32 1010, i32 20, !prof ![[PROF1:[0-9]]]
define i32 @LHSBin(i1 %C) !prof !0 {
  %A = select i1 %C, i32 1000, i32 10, !prof !1
  %V = add i32 %A, 10
  ret i32 %V
}

; CHECK: define i32 @RHSBin(i1 %C) !prof ![[PROF0]]
; CHECK: %V = select i1 %C, i32 1010, i32 20, !prof ![[PROF1]]
define i32 @RHSBin(i1 %C) !prof !0 {
  %A = select i1 %C, i32 1000, i32 10, !prof !1
  %V = add i32 10, %A
  ret i32 %V;
}

; CHECK: define i32 @BothBin(i1 %C) !prof ![[PROF0]]
; CHECK: %V = select i1 %C, i32 2000, i32 20, !prof ![[PROF1]]
define i32 @BothBin(i1 %C) !prof !0 {
  %A = select i1 %C, i32 1000, i32 10, !prof !1
  %B = select i1 %C, i32 1000, i32 10, !prof !1
  %V = add i32 %A, %B
  ret i32 %V;
}

; CHECK: define i32 @NegBin(i1 %C) !prof ![[PROF0:[0-9]]]
; CHECK: %V = select i1 %C, i32 1010, i32 0, !prof ![[PROF1]]
define i32 @NegBin(i1 %C) !prof !0 {
  %A = select i1 %C, i32 1000, i32 -10, !prof !1
  %V = add i32 %A, 10
  ret i32 %V
}

; CHECK: ![[PROF0]] = !{!"function_entry_count", i64 1000}
; CHECK: ![[PROF1]] = !{!"branch_weights", i32 2, i32 3}
!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 2, i32 3}
