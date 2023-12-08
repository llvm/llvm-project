; RUN: llc -mtriple=arm -float-abi=soft -mattr=+vfp2 %s -o - | FileCheck %s

define float @f1(float %a) {
; CHECK-LABEL: f1:
; CHECK: mov r0, #0
        ret float 0.000000e+00
}

define float @f2(ptr %v, float %u) {
; CHECK-LABEL: f2:
; CHECK: vldr{{.*}}[
        %tmp = load float, ptr %v           ; <float> [#uses=1]
        %tmp1 = fadd float %tmp, %u              ; <float> [#uses=1]
        ret float %tmp1
}

define float @f2offset(ptr %v, float %u) {
; CHECK-LABEL: f2offset:
; CHECK: vldr{{.*}}, #4]
        %addr = getelementptr float, ptr %v, i32 1
        %tmp = load float, ptr %addr
        %tmp1 = fadd float %tmp, %u
        ret float %tmp1
}

define float @f2noffset(ptr %v, float %u) {
; CHECK-LABEL: f2noffset:
; CHECK: vldr{{.*}}, #-4]
        %addr = getelementptr float, ptr %v, i32 -1
        %tmp = load float, ptr %addr
        %tmp1 = fadd float %tmp, %u
        ret float %tmp1
}

define void @f3(float %a, float %b, ptr %v) {
; CHECK-LABEL: f3:
; CHECK: vstr{{.*}}[
        %tmp = fadd float %a, %b         ; <float> [#uses=1]
        store float %tmp, ptr %v
        ret void
}
