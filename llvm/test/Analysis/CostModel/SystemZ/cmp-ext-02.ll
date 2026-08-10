; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=zEC12 | FileCheck %s
;
; Check the cost values for older subtargets that use an IPM sequence for
; extension of a compare result.

define i8 @fun0(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun0
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun1(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun1
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun2(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun2
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun3(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun3
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun4(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun4
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun5(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun5
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun6(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun6
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun7(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun7
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun8(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun8
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun9(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun9
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun10(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun10
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun11(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun11
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun12(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun12
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun13(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun13
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun14(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun14
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun15(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun15
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun16(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun16
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun17(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun17
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun18(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun18
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun19(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun19
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 5 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun20(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i8
  ret i8 %v

; CHECK: fun20
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i8
}

define i16 @fun21(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i16
  ret i16 %v

; CHECK: fun21
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i16
}

define i32 @fun22(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i32
  ret i32 %v

; CHECK: fun22
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = sext i1 %cmp to i32
}

define i64 @fun23(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = sext i1 %cmp to i64
  ret i64 %v

; CHECK: fun23
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 5 for instruction:   %v = sext i1 %cmp to i64
}

define i8 @fun120(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun120
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun121(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun121
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun122(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun122
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun123(i8 %val1, i8 %val2) {
  %cmp = icmp eq i8 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun123
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i8 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun124(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun124
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun125(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun125
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun126(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun126
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun127(i16 %val1, i16 %val2) {
  %cmp = icmp eq i16 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun127
; CHECK: cost of 3 for instruction:   %cmp = icmp eq i16 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun128(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun128
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun129(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun129
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun130(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun130
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun131(i32 %val1, i32 %val2) {
  %cmp = icmp eq i32 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun131
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i32 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun132(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun132
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun133(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun133
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun134(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun134
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun135(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun135
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 3 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun136(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun136
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun137(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun137
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun138(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun138
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun139(float %val1, float %val2) {
  %cmp = fcmp ogt float %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun139
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt float %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i64
}

define i8 @fun140(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i8
  ret i8 %v

; CHECK: fun140
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i8
}

define i16 @fun141(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i16
  ret i16 %v

; CHECK: fun141
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i16
}

define i32 @fun142(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i32
  ret i32 %v

; CHECK: fun142
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i32
}

define i64 @fun143(double %val1, double %val2) {
  %cmp = fcmp ogt double %val1, %val2
  %v = zext i1 %cmp to i64
  ret i64 %v

; CHECK: fun143
; CHECK: cost of 1 for instruction:   %cmp = fcmp ogt double %val1, %val2
; CHECK: cost of 4 for instruction:   %v = zext i1 %cmp to i64
}
