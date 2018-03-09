; RUN: llc < %s -mtriple=i686-- | FileCheck %s

define double @short(i16* %P) {
        %V = load i16, i16* %P               ; <i16> [#uses=1]
        %V2 = sitofp i16 %V to double           ; <double> [#uses=1]
        ret double %V2
}

define double @int(i32* %P) {
        %V = load i32, i32* %P               ; <i32> [#uses=1]
        %V2 = sitofp i32 %V to double           ; <double> [#uses=1]
        ret double %V2
}

define double @long(i64* %P) {
        %V = load i64, i64* %P               ; <i64> [#uses=1]
        %V2 = sitofp i64 %V to double           ; <double> [#uses=1]
        ret double %V2
}

; CHECK: long
; CHECK: fild
; CHECK-NOT: esp
; CHECK-NOT: esp
; CHECK: {{$}}
; CHECK: ret
