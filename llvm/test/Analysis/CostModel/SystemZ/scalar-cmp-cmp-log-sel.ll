; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; TODO: add more tests for differing operand types of the two compares.

define i8 @fun0(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun0
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun1(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                 i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun1
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun2(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                 i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun2
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun3(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                 i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun3
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun4(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                   float %val5, float %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun4
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun5(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                    double %val5, double %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun5
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun6(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun6
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun7(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                 i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun7
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun8(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                 i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun8
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun9(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                 i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun9
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun10(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun10
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun11(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun11
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun12(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun12
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun13(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun13
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun14(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun14
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun15(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun15
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun16(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun16
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun17(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun17
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun18(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun18
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun19(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun19
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun20(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun20
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun21(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun21
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun22(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun22
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun23(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun23
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun24(float %val1, float %val2, float %val3, float %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun24
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun25(float %val1, float %val2, float %val3, float %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun25
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun26(float %val1, float %val2, float %val3, float %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun26
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun27(float %val1, float %val2, float %val3, float %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun27
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun28(float %val1, float %val2, float %val3, float %val4,
                    float %val5, float %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun28
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun29(float %val1, float %val2, float %val3, float %val4,
                     double %val5, double %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun29
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun30(double %val1, double %val2, double %val3, double %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun30
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun31(double %val1, double %val2, double %val3, double %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun31
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun32(double %val1, double %val2, double %val3, double %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun32
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun33(double %val1, double %val2, double %val3, double %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun33
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun34(double %val1, double %val2, double %val3, double %val4,
                    float %val5, float %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun34
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun35(double %val1, double %val2, double %val3, double %val4,
                     double %val5, double %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = and i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun35
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = and i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun36(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun36
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun37(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun37
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun38(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun38
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun39(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun39
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun40(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun40
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun41(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun41
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun42(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun42
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun43(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun43
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun44(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun44
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun45(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun45
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun46(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun46
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun47(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun47
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun48(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun48
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun49(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun49
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun50(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun50
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun51(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun51
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun52(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun52
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun53(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun53
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun54(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun54
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun55(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun55
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun56(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun56
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun57(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun57
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun58(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun58
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun59(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun59
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun60(float %val1, float %val2, float %val3, float %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun60
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun61(float %val1, float %val2, float %val3, float %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun61
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun62(float %val1, float %val2, float %val3, float %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun62
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun63(float %val1, float %val2, float %val3, float %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun63
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun64(float %val1, float %val2, float %val3, float %val4,
                    float %val5, float %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun64
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun65(float %val1, float %val2, float %val3, float %val4,
                     double %val5, double %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun65
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun66(double %val1, double %val2, double %val3, double %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun66
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun67(double %val1, double %val2, double %val3, double %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun67
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun68(double %val1, double %val2, double %val3, double %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun68
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun69(double %val1, double %val2, double %val3, double %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun69
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun70(double %val1, double %val2, double %val3, double %val4,
                    float %val5, float %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun70
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun71(double %val1, double %val2, double %val3, double %val4,
                     double %val5, double %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = or i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun71
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 1 for instruction:   %and = or i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun72(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun72
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun73(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun73
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun74(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun74
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun75(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun75
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun76(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun76
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun77(i8 %val1, i8 %val2, i8 %val3, i8 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i8 %val1, %val2
  %cmp1 = icmp eq i8 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun77
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i8 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun78(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun78
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun79(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun79
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun80(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun80
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun81(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun81
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun82(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun82
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun83(i16 %val1, i16 %val2, i16 %val3, i16 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i16 %val1, %val2
  %cmp1 = icmp eq i16 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun83
; CHECK: cost of 3 for instruction:   %cmp0 = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %cmp1 = icmp eq i16 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun84(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun84
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun85(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun85
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun86(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun86
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun87(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun87
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun88(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun88
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun89(i32 %val1, i32 %val2, i32 %val3, i32 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i32 %val1, %val2
  %cmp1 = icmp eq i32 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun89
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i32 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun90(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun90
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun91(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun91
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun92(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun92
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun93(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun93
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun94(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                    float %val5, float %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun94
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun95(i64 %val1, i64 %val2, i64 %val3, i64 %val4,
                     double %val5, double %val6) {
  %cmp0 = icmp eq i64 %val1, %val2
  %cmp1 = icmp eq i64 %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun95
; CHECK: cost of 1 for instruction:   %cmp0 = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = icmp eq i64 %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun96(float %val1, float %val2, float %val3, float %val4,
                 i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun96
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun97(float %val1, float %val2, float %val3, float %val4,
                  i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun97
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun98(float %val1, float %val2, float %val3, float %val4,
                  i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun98
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun99(float %val1, float %val2, float %val3, float %val4,
                  i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun99
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun100(float %val1, float %val2, float %val3, float %val4,
                     float %val5, float %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun100
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun101(float %val1, float %val2, float %val3, float %val4,
                      double %val5, double %val6) {
  %cmp0 = fcmp ogt float %val1, %val2
  %cmp1 = fcmp ogt float %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun101
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt float %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

define i8 @fun102(double %val1, double %val2, double %val3, double %val4,
                  i8 %val5, i8 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i8 %val5, i8 %val6
  ret i8 %sel

; CHECK: fun102
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i8 %val5, i8 %val6
}

define i16 @fun103(double %val1, double %val2, double %val3, double %val4,
                   i16 %val5, i16 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i16 %val5, i16 %val6
  ret i16 %sel

; CHECK: fun103
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i16 %val5, i16 %val6
}

define i32 @fun104(double %val1, double %val2, double %val3, double %val4,
                   i32 %val5, i32 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i32 %val5, i32 %val6
  ret i32 %sel

; CHECK: fun104
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i32 %val5, i32 %val6
}

define i64 @fun105(double %val1, double %val2, double %val3, double %val4,
                   i64 %val5, i64 %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, i64 %val5, i64 %val6
  ret i64 %sel

; CHECK: fun105
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 1 for instruction:   %sel = select i1 %and, i64 %val5, i64 %val6
}

define float @fun106(double %val1, double %val2, double %val3, double %val4,
                     float %val5, float %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, float %val5, float %val6
  ret float %sel

; CHECK: fun106
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, float %val5, float %val6
}

define double @fun107(double %val1, double %val2, double %val3, double %val4,
                      double %val5, double %val6) {
  %cmp0 = fcmp ogt double %val1, %val2
  %cmp1 = fcmp ogt double %val3, %val4
  %and = xor i1 %cmp0, %cmp1
  %sel = select i1 %and, double %val5, double %val6
  ret double %sel

; CHECK: fun107
; CHECK: cost of 1 for instruction:   %cmp0 = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %cmp1 = fcmp ogt double %val3, %val4
; CHECK: cost of 7 for instruction:   %and = xor i1 %cmp0, %cmp1
; CHECK: cost of 4 for instruction:   %sel = select i1 %and, double %val5, double %val6
}

