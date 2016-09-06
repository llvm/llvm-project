; REQUIRES: asserts
; RUN: opt  < %s -passes='print<domfrontier>'  2>&1 | FileCheck %s

define void @a_linear_impl_fig_1() nounwind {
0:
  br label %"1"
1:
  br label %"2"
2:
  br label %"3"
3:
  br i1 1, label %"13", label %"4"
4:
  br i1 1, label %"5", label %"1"
5:
  br i1 1, label %"8", label %"6"
6:
  br i1 1, label %"7", label %"4"
7:
  ret void
8:
  br i1 1, label %"9", label %"1"
9:
  br label %"10"
10:
  br i1 1, label %"12", label %"11"
11:
  br i1 1, label %"9", label %"8"
13:
  br i1 1, label %"2", label %"1"
12:
   switch i32 0, label %"1" [ i32 0, label %"9"
                              i32 1, label %"8"]
}

; CHECK: DominanceFrontier for function: a_linear_impl_fig_1
; CHECK-DAG:  DomFrontier for BB %"0" is:
; CHECK-DAG:  DomFrontier for BB %"11" is:   %"{{[8|9]}}" %"{{[8|9]}}"
; CHECK-DAG:  DomFrontier for BB %"1" is:    %"1"
; CHECK-DAG:  DomFrontier for BB %"2" is:    %"{{[1|2]}}" %"{{[1|2]}}"
; CHECK-DAG:  DomFrontier for BB %"3" is:    %"{{[1|2]}}" %"{{[1|2]}}"
; CHECK-DAG:  DomFrontier for BB %"13" is:   %"{{[1|2]}}" %"{{[1|2]}}"
; CHECK-DAG:  DomFrontier for BB %"4" is:    %"{{[1|4]}}" %"{{[1|4]}}"
; CHECK-DAG:  DomFrontier for BB %"5" is:    %"{{[1|4]}}" %"{{[1|4]}}"
; CHECK-DAG:  DomFrontier for BB %"8" is:    %"{{[1|8]}}" %"{{[1|8]}}"
; CHECK-DAG:  DomFrontier for BB %"6" is:    %"4"
; CHECK-DAG:  DomFrontier for BB %"7" is:
; CHECK-DAG:  DomFrontier for BB %"9" is:    %"{{[1|8|9]}}" %"{{[1|8|9]}}" %"{{[1|8|9]}}"
; CHECK-DAG:  DomFrontier for BB %"10" is:   %"{{[1|8|9]}}" %"{{[1|8|9]}}" %"{{[1|8|9]}}"
; CHECK-DAG:  DomFrontier for BB %"12" is:   %"{{[1|8|9]}}" %"{{[1|8|9]}}" %"{{[1|8|9]}}"
