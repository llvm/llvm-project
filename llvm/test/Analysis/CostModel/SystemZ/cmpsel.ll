; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Note: Cost estimates of select of a fp-type is somewhat arbitrary, since it
; involves a conditional jump.
; Note: Vector fp32 is not directly supported, and not quite exact in
; estimates (but it is big absolute values).

define i8 @fun0(i8 %val1, i8 %val2,
                i8 %val3, i8 %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun0
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun1(i8 %val1, i8 %val2,
                 i16 %val3, i16 %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun1
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun2(i8 %val1, i8 %val2,
                 i32 %val3, i32 %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun2
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun3(i8 %val1, i8 %val2,
                 i64 %val3, i64 %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun3
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun4(i8 %val1, i8 %val2,
                   float %val3, float %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun4
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun5(i8 %val1, i8 %val2,
                    double %val3, double %val4) {
  %cmp = icmp eq i8 %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun5
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define i8 @fun6(i16 %val1, i16 %val2,
                i8 %val3, i8 %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun6
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun7(i16 %val1, i16 %val2,
                 i16 %val3, i16 %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun7
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun8(i16 %val1, i16 %val2,
                 i32 %val3, i32 %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun8
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun9(i16 %val1, i16 %val2,
                 i64 %val3, i64 %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun9
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun10(i16 %val1, i16 %val2,
                    float %val3, float %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun10
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun11(i16 %val1, i16 %val2,
                     double %val3, double %val4) {
  %cmp = icmp eq i16 %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun11
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define i8 @fun12(i32 %val1, i32 %val2,
                 i8 %val3, i8 %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun12
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun13(i32 %val1, i32 %val2,
                  i16 %val3, i16 %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun13
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun14(i32 %val1, i32 %val2,
                  i32 %val3, i32 %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun14
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun15(i32 %val1, i32 %val2,
                  i64 %val3, i64 %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun15
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun16(i32 %val1, i32 %val2,
                    float %val3, float %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun16
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun17(i32 %val1, i32 %val2,
                     double %val3, double %val4) {
  %cmp = icmp eq i32 %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun17
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define i8 @fun18(i64 %val1, i64 %val2,
                 i8 %val3, i8 %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun18
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun19(i64 %val1, i64 %val2,
                  i16 %val3, i16 %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun19
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun20(i64 %val1, i64 %val2,
                  i32 %val3, i32 %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun20
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun21(i64 %val1, i64 %val2,
                  i64 %val3, i64 %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun21
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun22(i64 %val1, i64 %val2,
                    float %val3, float %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun22
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun23(i64 %val1, i64 %val2,
                     double %val3, double %val4) {
  %cmp = icmp eq i64 %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun23
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define <2 x i8> @fun24(<2 x i8> %val1, <2 x i8> %val2,
                       <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun24
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun25(<2 x i8> %val1, <2 x i8> %val2,
                        <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun25
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun26(<2 x i8> %val1, <2 x i8> %val2,
                        <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun26
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun27(<2 x i8> %val1, <2 x i8> %val2,
                        <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun27
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun28(<2 x i8> %val1, <2 x i8> %val2,
                          <2 x float> %val3, <2 x float> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun28
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun29(<2 x i8> %val1, <2 x i8> %val2,
                           <2 x double> %val3, <2 x double> %val4) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun29
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <2 x i8> @fun30(<2 x i16> %val1, <2 x i16> %val2,
                       <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun30
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun31(<2 x i16> %val1, <2 x i16> %val2,
                        <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun31
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun32(<2 x i16> %val1, <2 x i16> %val2,
                        <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun32
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun33(<2 x i16> %val1, <2 x i16> %val2,
                        <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun33
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun34(<2 x i16> %val1, <2 x i16> %val2,
                          <2 x float> %val3, <2 x float> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun34
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun35(<2 x i16> %val1, <2 x i16> %val2,
                           <2 x double> %val3, <2 x double> %val4) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun35
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <2 x i8> @fun36(<2 x i32> %val1, <2 x i32> %val2,
                       <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun36
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun37(<2 x i32> %val1, <2 x i32> %val2,
                        <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun37
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun38(<2 x i32> %val1, <2 x i32> %val2,
                        <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun38
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun39(<2 x i32> %val1, <2 x i32> %val2,
                        <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun39
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun40(<2 x i32> %val1, <2 x i32> %val2,
                          <2 x float> %val3, <2 x float> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun40
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun41(<2 x i32> %val1, <2 x i32> %val2,
                           <2 x double> %val3, <2 x double> %val4) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun41
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <2 x i8> @fun42(<2 x i64> %val1, <2 x i64> %val2,
                       <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun42
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun43(<2 x i64> %val1, <2 x i64> %val2,
                        <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun43
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun44(<2 x i64> %val1, <2 x i64> %val2,
                        <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun44
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun45(<2 x i64> %val1, <2 x i64> %val2,
                        <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun45
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun46(<2 x i64> %val1, <2 x i64> %val2,
                          <2 x float> %val3, <2 x float> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun46
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun47(<2 x i64> %val1, <2 x i64> %val2,
                           <2 x double> %val3, <2 x double> %val4) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun47
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <4 x i8> @fun48(<4 x i8> %val1, <4 x i8> %val2,
                       <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun48
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun49(<4 x i8> %val1, <4 x i8> %val2,
                        <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun49
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun50(<4 x i8> %val1, <4 x i8> %val2,
                        <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun50
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun51(<4 x i8> %val1, <4 x i8> %val2,
                        <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun51
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 9 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun52(<4 x i8> %val1, <4 x i8> %val2,
                          <4 x float> %val3, <4 x float> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun52
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun53(<4 x i8> %val1, <4 x i8> %val2,
                           <4 x double> %val3, <4 x double> %val4) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun53
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 9 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <4 x i8> @fun54(<4 x i16> %val1, <4 x i16> %val2,
                       <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun54
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun55(<4 x i16> %val1, <4 x i16> %val2,
                        <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun55
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun56(<4 x i16> %val1, <4 x i16> %val2,
                        <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun56
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun57(<4 x i16> %val1, <4 x i16> %val2,
                        <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun57
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 7 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun58(<4 x i16> %val1, <4 x i16> %val2,
                          <4 x float> %val3, <4 x float> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun58
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun59(<4 x i16> %val1, <4 x i16> %val2,
                           <4 x double> %val3, <4 x double> %val4) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun59
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 7 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <4 x i8> @fun60(<4 x i32> %val1, <4 x i32> %val2,
                       <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun60
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun61(<4 x i32> %val1, <4 x i32> %val2,
                        <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun61
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun62(<4 x i32> %val1, <4 x i32> %val2,
                        <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun62
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun63(<4 x i32> %val1, <4 x i32> %val2,
                        <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun63
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun64(<4 x i32> %val1, <4 x i32> %val2,
                          <4 x float> %val3, <4 x float> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun64
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun65(<4 x i32> %val1, <4 x i32> %val2,
                           <4 x double> %val3, <4 x double> %val4) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun65
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <4 x i8> @fun66(<4 x i64> %val1, <4 x i64> %val2,
                       <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun66
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun67(<4 x i64> %val1, <4 x i64> %val2,
                        <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun67
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun68(<4 x i64> %val1, <4 x i64> %val2,
                        <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun68
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun69(<4 x i64> %val1, <4 x i64> %val2,
                        <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun69
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun70(<4 x i64> %val1, <4 x i64> %val2,
                          <4 x float> %val3, <4 x float> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun70
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun71(<4 x i64> %val1, <4 x i64> %val2,
                           <4 x double> %val3, <4 x double> %val4) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun71
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <8 x i8> @fun72(<8 x i8> %val1, <8 x i8> %val2,
                       <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun72
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun73(<8 x i8> %val1, <8 x i8> %val2,
                        <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun73
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun74(<8 x i8> %val1, <8 x i8> %val2,
                        <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun74
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 7 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun75(<8 x i8> %val1, <8 x i8> %val2,
                        <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun75
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 19 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun76(<8 x i8> %val1, <8 x i8> %val2,
                          <8 x float> %val3, <8 x float> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun76
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 7 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun77(<8 x i8> %val1, <8 x i8> %val2,
                           <8 x double> %val3, <8 x double> %val4) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun77
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 19 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <8 x i8> @fun78(<8 x i16> %val1, <8 x i16> %val2,
                       <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun78
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun79(<8 x i16> %val1, <8 x i16> %val2,
                        <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun79
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun80(<8 x i16> %val1, <8 x i16> %val2,
                        <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun80
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun81(<8 x i16> %val1, <8 x i16> %val2,
                        <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun81
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 15 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun82(<8 x i16> %val1, <8 x i16> %val2,
                          <8 x float> %val3, <8 x float> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun82
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun83(<8 x i16> %val1, <8 x i16> %val2,
                           <8 x double> %val3, <8 x double> %val4) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun83
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 15 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <8 x i8> @fun84(<8 x i32> %val1, <8 x i32> %val2,
                       <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun84
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun85(<8 x i32> %val1, <8 x i32> %val2,
                        <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun85
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun86(<8 x i32> %val1, <8 x i32> %val2,
                        <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun86
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun87(<8 x i32> %val1, <8 x i32> %val2,
                        <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun87
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun88(<8 x i32> %val1, <8 x i32> %val2,
                          <8 x float> %val3, <8 x float> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun88
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun89(<8 x i32> %val1, <8 x i32> %val2,
                           <8 x double> %val3, <8 x double> %val4) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun89
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <8 x i8> @fun90(<8 x i64> %val1, <8 x i64> %val2,
                       <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun90
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun91(<8 x i64> %val1, <8 x i64> %val2,
                        <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun91
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun92(<8 x i64> %val1, <8 x i64> %val2,
                        <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun92
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun93(<8 x i64> %val1, <8 x i64> %val2,
                        <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun93
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun94(<8 x i64> %val1, <8 x i64> %val2,
                          <8 x float> %val3, <8 x float> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun94
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun95(<8 x i64> %val1, <8 x i64> %val2,
                           <8 x double> %val3, <8 x double> %val4) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun95
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <16 x i8> @fun96(<16 x i8> %val1, <16 x i8> %val2,
                        <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun96
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun97(<16 x i8> %val1, <16 x i8> %val2,
                         <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun97
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun98(<16 x i8> %val1, <16 x i8> %val2,
                         <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun98
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 15 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun99(<16 x i8> %val1, <16 x i8> %val2,
                         <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun99
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 39 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun100(<16 x i8> %val1, <16 x i8> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun100
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 15 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun101(<16 x i8> %val1, <16 x i8> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun101
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 39 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

define <16 x i8> @fun102(<16 x i16> %val1, <16 x i16> %val2,
                         <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun102
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun103(<16 x i16> %val1, <16 x i16> %val2,
                          <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun103
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun104(<16 x i16> %val1, <16 x i16> %val2,
                          <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun104
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun105(<16 x i16> %val1, <16 x i16> %val2,
                          <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun105
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 31 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun106(<16 x i16> %val1, <16 x i16> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun106
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun107(<16 x i16> %val1, <16 x i16> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun107
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 31 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

define <16 x i8> @fun108(<16 x i32> %val1, <16 x i32> %val2,
                         <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun108
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun109(<16 x i32> %val1, <16 x i32> %val2,
                          <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun109
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun110(<16 x i32> %val1, <16 x i32> %val2,
                          <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun110
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun111(<16 x i32> %val1, <16 x i32> %val2,
                          <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun111
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 23 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun112(<16 x i32> %val1, <16 x i32> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun112
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun113(<16 x i32> %val1, <16 x i32> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun113
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 23 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

define <16 x i8> @fun114(<16 x i64> %val1, <16 x i64> %val2,
                         <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun114
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun115(<16 x i64> %val1, <16 x i64> %val2,
                          <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun115
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun116(<16 x i64> %val1, <16 x i64> %val2,
                          <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun116
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun117(<16 x i64> %val1, <16 x i64> %val2,
                          <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun117
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun118(<16 x i64> %val1, <16 x i64> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun118
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun119(<16 x i64> %val1, <16 x i64> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun119
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

define i8 @fun120(float %val1, float %val2,
                  i8 %val3, i8 %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun120
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun121(float %val1, float %val2,
                   i16 %val3, i16 %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun121
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun122(float %val1, float %val2,
                   i32 %val3, i32 %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun122
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun123(float %val1, float %val2,
                   i64 %val3, i64 %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun123
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun124(float %val1, float %val2,
                     float %val3, float %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun124
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun125(float %val1, float %val2,
                      double %val3, double %val4) {
  %cmp = fcmp ogt float %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun125
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define i8 @fun126(double %val1, double %val2,
                  i8 %val3, i8 %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, i8 %val3, i8 %val4
  ret i8 %sel

; CHECK: fun126
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i8 %val3, i8 %val4
}

define i16 @fun127(double %val1, double %val2,
                   i16 %val3, i16 %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, i16 %val3, i16 %val4
  ret i16 %sel

; CHECK: fun127
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i16 %val3, i16 %val4
}

define i32 @fun128(double %val1, double %val2,
                   i32 %val3, i32 %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, i32 %val3, i32 %val4
  ret i32 %sel

; CHECK: fun128
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i32 %val3, i32 %val4
}

define i64 @fun129(double %val1, double %val2,
                   i64 %val3, i64 %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, i64 %val3, i64 %val4
  ret i64 %sel

; CHECK: fun129
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select i1 %cmp, i64 %val3, i64 %val4
}

define float @fun130(double %val1, double %val2,
                     float %val3, float %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, float %val3, float %val4
  ret float %sel

; CHECK: fun130
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, float %val3, float %val4
}

define double @fun131(double %val1, double %val2,
                      double %val3, double %val4) {
  %cmp = fcmp ogt double %val1, %val2
  %sel = select i1 %cmp, double %val3, double %val4
  ret double %sel

; CHECK: fun131
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select i1 %cmp, double %val3, double %val4
}

define <2 x i8> @fun132(<2 x float> %val1, <2 x float> %val2,
                        <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun132
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun133(<2 x float> %val1, <2 x float> %val2,
                         <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun133
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun134(<2 x float> %val1, <2 x float> %val2,
                         <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun134
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun135(<2 x float> %val1, <2 x float> %val2,
                         <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun135
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun136(<2 x float> %val1, <2 x float> %val2,
                           <2 x float> %val3, <2 x float> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun136
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun137(<2 x float> %val1, <2 x float> %val2,
                            <2 x double> %val3, <2 x double> %val4) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun137
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <2 x i8> @fun138(<2 x double> %val1, <2 x double> %val2,
                        <2 x i8> %val3, <2 x i8> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
  ret <2 x i8> %sel

; CHECK: fun138
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i8> %val3, <2 x i8> %val4
}

define <2 x i16> @fun139(<2 x double> %val1, <2 x double> %val2,
                         <2 x i16> %val3, <2 x i16> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
  ret <2 x i16> %sel

; CHECK: fun139
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i16> %val3, <2 x i16> %val4
}

define <2 x i32> @fun140(<2 x double> %val1, <2 x double> %val2,
                         <2 x i32> %val3, <2 x i32> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
  ret <2 x i32> %sel

; CHECK: fun140
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x i32> %val3, <2 x i32> %val4
}

define <2 x i64> @fun141(<2 x double> %val1, <2 x double> %val2,
                         <2 x i64> %val3, <2 x i64> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
  ret <2 x i64> %sel

; CHECK: fun141
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x i64> %val3, <2 x i64> %val4
}

define <2 x float> @fun142(<2 x double> %val1, <2 x double> %val2,
                           <2 x float> %val3, <2 x float> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
  ret <2 x float> %sel

; CHECK: fun142
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <2 x i1> %cmp, <2 x float> %val3, <2 x float> %val4
}

define <2 x double> @fun143(<2 x double> %val1, <2 x double> %val2,
                            <2 x double> %val3, <2 x double> %val4) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
  ret <2 x double> %sel

; CHECK: fun143
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <2 x i1> %cmp, <2 x double> %val3, <2 x double> %val4
}

define <4 x i8> @fun144(<4 x float> %val1, <4 x float> %val2,
                        <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun144
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun145(<4 x float> %val1, <4 x float> %val2,
                         <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun145
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun146(<4 x float> %val1, <4 x float> %val2,
                         <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun146
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun147(<4 x float> %val1, <4 x float> %val2,
                         <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun147
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun148(<4 x float> %val1, <4 x float> %val2,
                           <4 x float> %val3, <4 x float> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun148
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun149(<4 x float> %val1, <4 x float> %val2,
                            <4 x double> %val3, <4 x double> %val4) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun149
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 5 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <4 x i8> @fun150(<4 x double> %val1, <4 x double> %val2,
                        <4 x i8> %val3, <4 x i8> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
  ret <4 x i8> %sel

; CHECK: fun150
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i8> %val3, <4 x i8> %val4
}

define <4 x i16> @fun151(<4 x double> %val1, <4 x double> %val2,
                         <4 x i16> %val3, <4 x i16> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
  ret <4 x i16> %sel

; CHECK: fun151
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i16> %val3, <4 x i16> %val4
}

define <4 x i32> @fun152(<4 x double> %val1, <4 x double> %val2,
                         <4 x i32> %val3, <4 x i32> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
  ret <4 x i32> %sel

; CHECK: fun152
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i32> %val3, <4 x i32> %val4
}

define <4 x i64> @fun153(<4 x double> %val1, <4 x double> %val2,
                         <4 x i64> %val3, <4 x i64> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
  ret <4 x i64> %sel

; CHECK: fun153
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x i64> %val3, <4 x i64> %val4
}

define <4 x float> @fun154(<4 x double> %val1, <4 x double> %val2,
                           <4 x float> %val3, <4 x float> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
  ret <4 x float> %sel

; CHECK: fun154
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x float> %val3, <4 x float> %val4
}

define <4 x double> @fun155(<4 x double> %val1, <4 x double> %val2,
                            <4 x double> %val3, <4 x double> %val4) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
  ret <4 x double> %sel

; CHECK: fun155
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <4 x i1> %cmp, <4 x double> %val3, <4 x double> %val4
}

define <8 x i8> @fun156(<8 x float> %val1, <8 x float> %val2,
                        <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun156
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun157(<8 x float> %val1, <8 x float> %val2,
                         <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun157
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun158(<8 x float> %val1, <8 x float> %val2,
                         <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun158
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun159(<8 x float> %val1, <8 x float> %val2,
                         <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun159
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun160(<8 x float> %val1, <8 x float> %val2,
                           <8 x float> %val3, <8 x float> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun160
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun161(<8 x float> %val1, <8 x float> %val2,
                            <8 x double> %val3, <8 x double> %val4) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun161
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 11 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <8 x i8> @fun162(<8 x double> %val1, <8 x double> %val2,
                        <8 x i8> %val3, <8 x i8> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
  ret <8 x i8> %sel

; CHECK: fun162
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i8> %val3, <8 x i8> %val4
}

define <8 x i16> @fun163(<8 x double> %val1, <8 x double> %val2,
                         <8 x i16> %val3, <8 x i16> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
  ret <8 x i16> %sel

; CHECK: fun163
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i16> %val3, <8 x i16> %val4
}

define <8 x i32> @fun164(<8 x double> %val1, <8 x double> %val2,
                         <8 x i32> %val3, <8 x i32> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
  ret <8 x i32> %sel

; CHECK: fun164
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i32> %val3, <8 x i32> %val4
}

define <8 x i64> @fun165(<8 x double> %val1, <8 x double> %val2,
                         <8 x i64> %val3, <8 x i64> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
  ret <8 x i64> %sel

; CHECK: fun165
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x i64> %val3, <8 x i64> %val4
}

define <8 x float> @fun166(<8 x double> %val1, <8 x double> %val2,
                           <8 x float> %val3, <8 x float> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
  ret <8 x float> %sel

; CHECK: fun166
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x float> %val3, <8 x float> %val4
}

define <8 x double> @fun167(<8 x double> %val1, <8 x double> %val2,
                            <8 x double> %val3, <8 x double> %val4) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
  ret <8 x double> %sel

; CHECK: fun167
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <8 x i1> %cmp, <8 x double> %val3, <8 x double> %val4
}

define <16 x i8> @fun168(<16 x float> %val1, <16 x float> %val2,
                         <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun168
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun169(<16 x float> %val1, <16 x float> %val2,
                          <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun169
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun170(<16 x float> %val1, <16 x float> %val2,
                          <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun170
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun171(<16 x float> %val1, <16 x float> %val2,
                          <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun171
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 23 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun172(<16 x float> %val1, <16 x float> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun172
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun173(<16 x float> %val1, <16 x float> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun173
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 23 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

define <16 x i8> @fun174(<16 x double> %val1, <16 x double> %val2,
                         <16 x i8> %val3, <16 x i8> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
  ret <16 x i8> %sel

; CHECK: fun174
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i8> %val3, <16 x i8> %val4
}

define <16 x i16> @fun175(<16 x double> %val1, <16 x double> %val2,
                          <16 x i16> %val3, <16 x i16> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
  ret <16 x i16> %sel

; CHECK: fun175
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i16> %val3, <16 x i16> %val4
}

define <16 x i32> @fun176(<16 x double> %val1, <16 x double> %val2,
                          <16 x i32> %val3, <16 x i32> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
  ret <16 x i32> %sel

; CHECK: fun176
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i32> %val3, <16 x i32> %val4
}

define <16 x i64> @fun177(<16 x double> %val1, <16 x double> %val2,
                          <16 x i64> %val3, <16 x i64> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
  ret <16 x i64> %sel

; CHECK: fun177
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x i64> %val3, <16 x i64> %val4
}

define <16 x float> @fun178(<16 x double> %val1, <16 x double> %val2,
                            <16 x float> %val3, <16 x float> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
  ret <16 x float> %sel

; CHECK: fun178
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x float> %val3, <16 x float> %val4
}

define <16 x double> @fun179(<16 x double> %val1, <16 x double> %val2,
                             <16 x double> %val3, <16 x double> %val4) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
  ret <16 x double> %sel

; CHECK: fun179
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %sel = select <16 x i1> %cmp, <16 x double> %val3, <16 x double> %val4
}

