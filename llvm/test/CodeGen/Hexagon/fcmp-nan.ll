; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Test that all FP ordered compare instructions generate the correct
; post-processing to accommodate NaNs.
;
; Specifically for ordered FP compares, we have to check if one of
; the operands was a NaN to comform to the semantics of the ordered
; fcmp bitcode instruction
;
target triple = "hexagon"

;
; Functions for float:
;

;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.eq(r0,r1)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r0,r1)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_oeq_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp oeq float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.eq(r0,r1)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r0,r1)
; CHECK: [[REG2:p([0-3])]] = or([[REG0]],[[REG1]])
; CHECK: r0 = mux([[REG2]],#0,#1)
;
define i32 @compare_one_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp one float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.gt(r0,r1)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r0,r1)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_ogt_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp ogt float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.ge(r1,r0)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r1,r0)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_ole_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp ole float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}



;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.ge(r0,r1)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r0,r1)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_oge_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp oge float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = sfcmp.gt(r1,r0)
; CHECK-DAG: [[REG1:p([0-3])]] = sfcmp.uo(r1,r0)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_olt_f(float %val, float %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp olt float %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}



;
; Functions for double:
;

;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.eq(r1:0,r3:2)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r1:0,r3:2)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_oeq_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp oeq double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.eq(r1:0,r3:2)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r1:0,r3:2)
; CHECK: [[REG2:p([0-3])]] = or([[REG0]],[[REG1]])
; CHECK: r0 = mux([[REG2]],#0,#1)
;
define i32 @compare_one_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp one double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}



;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.gt(r1:0,r3:2)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r1:0,r3:2)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_ogt_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp ogt double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.ge(r3:2,r1:0)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r3:2,r1:0)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_ole_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp ole double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.ge(r1:0,r3:2)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r1:0,r3:2)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_oge_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp oge double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}


;
; CHECK-DAG: [[REG0:p([0-3])]] = dfcmp.gt(r3:2,r1:0)
; CHECK-DAG: [[REG1:p([0-3])]] = dfcmp.uo(r3:2,r1:0)
; CHECK: [[REG2:p([0-3])]] = and([[REG0]],![[REG1]])
; CHECK: r0 = mux([[REG2]],#1,#0)
;
define i32 @compare_olt_d(double %val, double %val2) local_unnamed_addr #0 {
entry:
  %cmpinf = fcmp olt double %val, %val2
  %0 = zext i1 %cmpinf to i32
  ret i32 %0
}

