; REQUIRES: asserts
; RUN: opt  < %s -passes='print<domfrontier>'  2>&1 | FileCheck %s

define void @a_linear_impl_fig_1() nounwind {
0:
  br label %1
1:
  br label %2
2:
  br label %3
3:
  br i1 1, label %a12, label %4
4:
  br i1 1, label %5, label %1
5:
  br i1 1, label %a8, label %6
6:
  br i1 1, label %a7, label %4
a7:
  ret void
a8:
  br i1 1, label %a9, label %1
a9:
  br label %a10
a10:
  br i1 1, label %a13, label %a11
a11:
  br i1 1, label %a9, label %a8
a12:
  br i1 1, label %2, label %1
a13:
   switch i32 0, label %1 [ i32 0, label %a9
                              i32 1, label %a8]
}

; CHECK: DominanceFrontier for function: a_linear_impl_fig_1
; CHECK-DAG:  DomFrontier for BB %0 is:
; CHECK-DAG:  DomFrontier for BB %a11 is:   %a9 %a8
; CHECK-DAG:  DomFrontier for BB %1 is:    %1
; CHECK-DAG:  DomFrontier for BB %2 is:    %1 %2
; CHECK-DAG:  DomFrontier for BB %3 is:    %1 %2
; CHECK-DAG:  DomFrontier for BB %a12 is:   %2 %1
; CHECK-DAG:  DomFrontier for BB %4 is:    %1 %4
; CHECK-DAG:  DomFrontier for BB %5 is:    %4 %1
; CHECK-DAG:  DomFrontier for BB %a8 is:    %1 %a8
; CHECK-DAG:  DomFrontier for BB %6 is:    %4
; CHECK-DAG:  DomFrontier for BB %a7 is:
; CHECK-DAG:  DomFrontier for BB %a9 is:    %a9 %a8 %1
; CHECK-DAG:  DomFrontier for BB %a10 is:   %a9 %a8 %1
; CHECK-DAG:  DomFrontier for BB %a13 is:   %1 %a9 %a8
