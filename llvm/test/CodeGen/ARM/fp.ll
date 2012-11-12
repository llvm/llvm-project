; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s

define float @f(i32 %a) {
;CHECK: f:
;CHECK: vmov
;CHECK-NEXT: vcvt.f32.s32
;CHECK-NEXT: vmov
entry:
        %tmp = sitofp i32 %a to float           ; <float> [#uses=1]
        ret float %tmp
}

define double @g(i32 %a) {
;CHECK: g:
;CHECK: vmov
;CHECK-NEXT: vcvt.f64.s32
;CHECK-NEXT: vmov
entry:
        %tmp = sitofp i32 %a to double          ; <double> [#uses=1]
        ret double %tmp
}

define double @uint_to_double(i32 %a) {
;CHECK: uint_to_double:
;CHECK: vmov
;CHECK-NEXT: vcvt.f64.u32
;CHECK-NEXT: vmov
entry:
        %tmp = uitofp i32 %a to double          ; <double> [#uses=1]
        ret double %tmp
}

define float @uint_to_float(i32 %a) {
;CHECK: uint_to_float:
;CHECK: vmov
;CHECK-NEXT: vcvt.f32.u32
;CHECK-NEXT: vmov
entry:
        %tmp = uitofp i32 %a to float           ; <float> [#uses=1]
        ret float %tmp
}

define double @h(double* %v) {
;CHECK: h:
;CHECK: vldr
;CHECK-NEXT: vmov
entry:
        %tmp = load double* %v          ; <double> [#uses=1]
        ret double %tmp
}

define float @h2() {
;CHECK: h2:
;CHECK: mov r0, #1065353216
entry:
        ret float 1.000000e+00
}

define double @f2(double %a) {
;CHECK: f2:
;CHECK-NOT: vmov
        ret double %a
}

define void @f3() {
;CHECK: f3:
;CHECK-NOT: vmov
;CHECK: f4
entry:
        %tmp = call double @f5( )               ; <double> [#uses=1]
        call void @f4( double %tmp )
        ret void
}

declare void @f4(double)

declare double @f5()

