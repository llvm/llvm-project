; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - | FileCheck %s

define i32 @f1(float %a) {
;CHECK-LABEL: f1:
;CHECK: vcmp.f32
;CHECK: movmi
entry:
        %tmp = fcmp olt float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp1 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(float %a) {
;CHECK-LABEL: f2:
;CHECK: vcmp.f32
;CHECK: moveq
entry:
        %tmp = fcmp oeq float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp2 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp2
}

define i32 @f3(float %a) {
;CHECK-LABEL: f3:
;CHECK: vcmp.f32
;CHECK: movgt
entry:
        %tmp = fcmp ogt float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp3 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp3
}

define i32 @f4(float %a) {
;CHECK-LABEL: f4:
;CHECK: vcmp.f32
;CHECK: movge
entry:
        %tmp = fcmp oge float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp4 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f5(float %a) {
;CHECK-LABEL: f5:
;CHECK: vcmp.f32
;CHECK: movls
entry:
        %tmp = fcmp ole float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp5 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp5
}

define i32 @f6(float %a) {
;CHECK-LABEL: f6:
;CHECK: vcmp.f32
;CHECK: movne
entry:
        %tmp = fcmp une float %a, 1.000000e+00          ; <i1> [#uses=1]
        %tmp6 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp6
}

define i32 @g1(double %a) {
;CHECK-LABEL: g1:
;CHECK: vcmp.f64
;CHECK: movmi
entry:
        %tmp = fcmp olt double %a, 1.000000e+00         ; <i1> [#uses=1]
        %tmp7 = zext i1 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp7
}
