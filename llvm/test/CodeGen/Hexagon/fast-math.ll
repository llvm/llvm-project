; RUN: llc -march=hexagon -mcpu=hexagonv65 -fast-math < %s | FileCheck %s --check-prefix=MATHV5

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define double @ffoo0(double %a, double %b, double %c) nounwind readnone {
entry:
; MATH5: __hexagon_fast2_muldf3

  %mul = fmul double %a, %c
  ret double %mul
}

define double @ffoo1(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: __hexagon_fast2_subdf3
  %sub = fsub double %a, %c
  ret double %sub
}

define double @ffoo2(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: __hexagon_fast2_divdf3
  %div = fdiv double %a, %c
  ret double %div
}

define double @ffoo3(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: __hexagon_fast2_adddf3
  %add = fadd double %a, %c
  ret double %add
}

define double @ffoo4(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: togglebit
  %sub = fsub double -0.000000e+00, %c
  ret double %sub
}

define double @ffoo5b(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: __hexagon_fast2_sqrtdf2
  %call = tail call double @sqrt(double %c) nounwind readnone
  ret double %call
}

declare double @sqrt(double) nounwind readnone

define double @ffoo5c(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5: __hexagon_fast2_sqrtdf2
  %call = tail call double @sqrt(double %c)
  ret double %call
}

define i32 @ffoo6(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_ltdf2
  %cmp = fcmp olt double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @ffoo7(double %a, double %b, double %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_gtdf2
  %cmp = fcmp ogt double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @ffoo8(double %a, double %b, double %c) nounwind readnone {
entry:
  %cmp = fcmp ole double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @ffoo9(double %a, double %b, double %c) nounwind readnone {
entry:
  %cmp = fcmp oge double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @ffoo10(double %a, double %b, double %c) nounwind readnone {
entry:
  %cmp = fcmp oeq double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @ffoo11(double %a, double %b, double %c) nounwind readnone {
entry:
  %cmp = fcmp une double %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define float @fgoo0(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_mulsf3
  %mul = fmul float %a, %c
  ret float %mul
}

define float @fgoo1(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_subsf3
  %sub = fsub float %a, %c
  ret float %sub
}

define float @fgoo2(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5: += sfmpy({{.*}}):lib
; MATHV5: -= sfmpy({{.*}}):lib
; MATHV5: -= sfmpy({{.*}}):lib
; MATHV5: += sfmpy({{.*}}):lib
; MATHV5: += sfmpy({{.*}}):lib
; MATHV5: -= sfmpy({{.*}}):lib
; MATHV5: += sfmpy{{.*}}:scale
  %div = fdiv float %a, %c
  ret float %div
}

define float @fgoo3(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_addsf3
  %add = fadd float %a, %c
  ret float %add
}

define float @fgoo4(float %a, float %b, float %c) nounwind readnone {
entry:
  %sub = fsub float -0.000000e+00, %c
  ret float %sub
}

define float @fgoo5b(float %a, float %b, float %c) nounwind readnone {
entry:
  %call = tail call float @sqrtf(float %c) nounwind readnone
  ret float %call
}

declare float @sqrtf(float) nounwind readnone

define float @fgoo5c(float %a, float %b, float %c) nounwind readnone {
entry:
  %call = tail call float @sqrtf(float %c)
  ret float %call
}

define i32 @fgoo6(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_ltsf2
  %cmp = fcmp olt float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @fgoo7(float %a, float %b, float %c) nounwind readnone {
entry:
; MATHV5-NOT: __hexagon_fast2_gtsf2
  %cmp = fcmp ogt float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @fgoo8(float %a, float %b, float %c) nounwind readnone {
entry:
  %cmp = fcmp ole float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @fgoo9(float %a, float %b, float %c) nounwind readnone {
entry:
  %cmp = fcmp oge float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @fgoo10(float %a, float %b, float %c) nounwind readnone {
entry:
  %cmp = fcmp oeq float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @fgoo11(float %a, float %b, float %c) nounwind readnone {
entry:
  %cmp = fcmp une float %a, %c
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
