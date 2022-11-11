; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;

define i8 @fun0(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun0
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun1(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun1
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun2(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun2
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun3(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun3
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun4(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun4
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun5(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun5
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun6(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun6
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun7(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun7
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun8(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun8
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun9(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun9
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun10(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun10
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun11(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun11
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun12(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun12
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun13(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun13
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun14(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun14
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun15(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun15
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun16(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun16
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun17(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun17
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun18(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun18
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun19(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun19
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun20(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun20
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun21(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun21
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun22(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun22
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun23(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun23
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext i1 %cmp to i64
}

define <2 x i8> @fun24(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun24
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun25(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun25
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun26(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun26
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun27(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun27
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun28(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun28
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun29(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun29
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun30(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun30
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun31(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun31
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun32(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun32
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun33(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun33
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun34(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun34
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun35(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun35
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun36(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun36
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun37(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun37
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun38(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun38
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun39(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun39
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun40(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun40
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun41(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun41
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun42(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun42
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun43(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun43
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun44(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun44
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun45(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun45
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun46(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun46
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun47(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun47
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <2 x i1> %cmp to <2 x i64>
}

define <4 x i8> @fun48(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun48
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun49(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun49
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun50(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun50
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun51(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun51
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun52(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun52
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun53(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun53
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun54(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun54
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun55(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun55
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun56(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun56
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun57(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun57
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun58(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun58
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun59(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun59
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun60(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun60
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun61(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun61
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun62(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun62
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun63(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun63
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun64(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun64
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun65(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun65
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun66(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun66
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun67(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun67
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun68(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun68
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun69(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun69
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun70(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun70
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun71(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = sext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun71
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <4 x i1> %cmp to <4 x i64>
}

define <8 x i8> @fun72(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun72
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun73(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun73
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun74(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun74
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun75(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun75
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 15 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun76(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun76
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun77(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun77
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun78(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun78
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun79(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun79
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 11 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun80(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun80
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun81(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun81
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun82(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun82
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun83(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun83
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun84(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun84
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun85(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun85
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun86(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun86
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun87(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun87
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun88(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun88
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun89(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun89
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun90(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun90
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun91(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun91
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun92(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun92
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun93(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun93
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun94(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun94
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun95(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = sext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun95
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <8 x i1> %cmp to <8 x i64>
}

define <16 x i8> @fun96(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun96
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun97(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun97
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun98(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun98
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 11 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun99(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun99
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 31 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun100(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun100
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun101(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun101
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun102(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun102
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun103(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun103
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 23 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun104(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun104
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun105(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun105
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun106(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun106
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun107(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun107
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 15 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun108(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun108
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun109(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun109
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 6 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun110(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun110
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun111(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun111
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun112(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun112
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun113(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun113
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun114(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun114
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun115(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun115
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 15 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun116(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun116
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = sext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun117(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun117
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 6 for instruction:   %v = sext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun118(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun118
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun119(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = sext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun119
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 0 for instruction:   %v = sext <16 x i1> %cmp to <16 x i64>
}

define i8 @fun120(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun120
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun121(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun121
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun122(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun122
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun123(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun123
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun124(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun124
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun125(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun125
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun126(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun126
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun127(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun127
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun128(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun128
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun129(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun129
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun130(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun130
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun131(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun131
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun132(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun132
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun133(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun133
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun134(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun134
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun135(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun135
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun136(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun136
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun137(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun137
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun138(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun138
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun139(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun139
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun140(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun140
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun141(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun141
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun142(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun142
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun143(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun143
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext i1 %cmp to i64
}

define <2 x i8> @fun144(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun144
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun145(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun145
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun146(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun146
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun147(<2 x i8> %val1, <2 x i8> %val2) {
  %cmp = icmp eq <2 x i8> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun147
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i8> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun148(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun148
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun149(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun149
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun150(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun150
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun151(<2 x i16> %val1, <2 x i16> %val2) {
  %cmp = icmp eq <2 x i16> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun151
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i16> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun152(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun152
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun153(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun153
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun154(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun154
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun155(<2 x i32> %val1, <2 x i32> %val2) {
  %cmp = icmp eq <2 x i32> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun155
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun156(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun156
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun157(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun157
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun158(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun158
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun159(<2 x i64> %val1, <2 x i64> %val2) {
  %cmp = icmp eq <2 x i64> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun159
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <2 x i64> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun160(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun160
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun161(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun161
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun162(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun162
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun163(<2 x float> %val1, <2 x float> %val2) {
  %cmp = fcmp ogt <2 x float> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun163
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <2 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <2 x i8> @fun164(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %v

; CHECK: fun164
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i8>
}

define <2 x i16> @fun165(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %v

; CHECK: fun165
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i16>
}

define <2 x i32> @fun166(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %v

; CHECK: fun166
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <2 x i1> %cmp to <2 x i32>
}

define <2 x i64> @fun167(<2 x double> %val1, <2 x double> %val2) {
  %cmp = fcmp ogt <2 x double> %val1, %val2
  %v = zext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %v

; CHECK: fun167
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt <2 x double> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <2 x i1> %cmp to <2 x i64>
}

define <4 x i8> @fun168(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun168
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun169(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun169
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun170(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun170
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun171(<4 x i8> %val1, <4 x i8> %val2) {
  %cmp = icmp eq <4 x i8> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun171
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i8> %val1, %val2
; CHECK: cost of 9 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun172(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun172
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun173(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun173
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun174(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun174
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun175(<4 x i16> %val1, <4 x i16> %val2) {
  %cmp = icmp eq <4 x i16> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun175
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i16> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun176(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun176
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun177(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun177
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun178(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun178
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun179(<4 x i32> %val1, <4 x i32> %val2) {
  %cmp = icmp eq <4 x i32> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun179
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <4 x i32> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun180(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun180
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun181(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun181
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun182(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun182
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun183(<4 x i64> %val1, <4 x i64> %val2) {
  %cmp = icmp eq <4 x i64> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun183
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <4 x i64> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun184(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun184
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun185(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun185
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun186(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun186
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun187(<4 x float> %val1, <4 x float> %val2) {
  %cmp = fcmp ogt <4 x float> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun187
; CHECK: cost of 10 for instruction:   %cmp = fcmp ogt <4 x float> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <4 x i8> @fun188(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %v

; CHECK: fun188
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i8>
}

define <4 x i16> @fun189(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %v

; CHECK: fun189
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i16>
}

define <4 x i32> @fun190(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %v

; CHECK: fun190
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i32>
}

define <4 x i64> @fun191(<4 x double> %val1, <4 x double> %val2) {
  %cmp = fcmp ogt <4 x double> %val1, %val2
  %v = zext <4 x i1> %cmp to <4 x i64>
  ret <4 x i64> %v

; CHECK: fun191
; CHECK: cost of 2 for instruction:   %cmp = fcmp ogt <4 x double> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <4 x i1> %cmp to <4 x i64>
}

define <8 x i8> @fun192(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun192
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun193(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun193
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun194(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun194
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 7 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun195(<8 x i8> %val1, <8 x i8> %val2) {
  %cmp = icmp eq <8 x i8> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun195
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i8> %val1, %val2
; CHECK: cost of 19 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun196(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun196
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun197(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun197
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun198(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun198
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun199(<8 x i16> %val1, <8 x i16> %val2) {
  %cmp = icmp eq <8 x i16> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun199
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <8 x i16> %val1, %val2
; CHECK: cost of 15 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun200(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun200
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun201(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun201
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun202(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun202
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun203(<8 x i32> %val1, <8 x i32> %val2) {
  %cmp = icmp eq <8 x i32> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun203
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <8 x i32> %val1, %val2
; CHECK: cost of 11 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun204(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun204
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun205(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun205
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun206(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun206
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun207(<8 x i64> %val1, <8 x i64> %val2) {
  %cmp = icmp eq <8 x i64> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun207
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <8 x i64> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun208(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun208
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun209(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun209
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun210(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun210
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun211(<8 x float> %val1, <8 x float> %val2) {
  %cmp = fcmp ogt <8 x float> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun211
; CHECK: cost of 20 for instruction:   %cmp = fcmp ogt <8 x float> %val1, %val2
; CHECK: cost of 11 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <8 x i8> @fun212(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %v

; CHECK: fun212
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i8>
}

define <8 x i16> @fun213(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %v

; CHECK: fun213
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i16>
}

define <8 x i32> @fun214(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %v

; CHECK: fun214
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i32>
}

define <8 x i64> @fun215(<8 x double> %val1, <8 x double> %val2) {
  %cmp = fcmp ogt <8 x double> %val1, %val2
  %v = zext <8 x i1> %cmp to <8 x i64>
  ret <8 x i64> %v

; CHECK: fun215
; CHECK: cost of 4 for instruction:   %cmp = fcmp ogt <8 x double> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <8 x i1> %cmp to <8 x i64>
}

define <16 x i8> @fun216(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun216
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 1 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun217(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun217
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 5 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun218(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun218
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 15 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun219(<16 x i8> %val1, <16 x i8> %val2) {
  %cmp = icmp eq <16 x i8> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun219
; CHECK: cost of 1 for instruction:   %cmp = icmp eq <16 x i8> %val1, %val2
; CHECK: cost of 39 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun220(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun220
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun221(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun221
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 2 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun222(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun222
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 11 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun223(<16 x i16> %val1, <16 x i16> %val2) {
  %cmp = icmp eq <16 x i16> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun223
; CHECK: cost of 2 for instruction:   %cmp = icmp eq <16 x i16> %val1, %val2
; CHECK: cost of 31 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun224(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun224
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun225(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun225
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun226(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun226
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun227(<16 x i32> %val1, <16 x i32> %val2) {
  %cmp = icmp eq <16 x i32> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun227
; CHECK: cost of 4 for instruction:   %cmp = icmp eq <16 x i32> %val1, %val2
; CHECK: cost of 23 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun228(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun228
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun229(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun229
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun230(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun230
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun231(<16 x i64> %val1, <16 x i64> %val2) {
  %cmp = icmp eq <16 x i64> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun231
; CHECK: cost of 8 for instruction:   %cmp = icmp eq <16 x i64> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun232(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun232
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun233(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun233
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun234(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun234
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun235(<16 x float> %val1, <16 x float> %val2) {
  %cmp = fcmp ogt <16 x float> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun235
; CHECK: cost of 40 for instruction:   %cmp = fcmp ogt <16 x float> %val1, %val2
; CHECK: cost of 23 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

define <16 x i8> @fun236(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %v

; CHECK: fun236
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i8>
}

define <16 x i16> @fun237(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %v

; CHECK: fun237
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i16>
}

define <16 x i32> @fun238(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %v

; CHECK: fun238
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i32>
}

define <16 x i64> @fun239(<16 x double> %val1, <16 x double> %val2) {
  %cmp = fcmp ogt <16 x double> %val1, %val2
  %v = zext <16 x i1> %cmp to <16 x i64>
  ret <16 x i64> %v

; CHECK: fun239
; CHECK: cost of 8 for instruction:   %cmp = fcmp ogt <16 x double> %val1, %val2
; CHECK: cost of 8 for instruction:   %v = zext <16 x i1> %cmp to <16 x i64>
}

