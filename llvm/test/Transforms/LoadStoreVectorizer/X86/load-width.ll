; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -mcpu haswell -S -o - %s | FileCheck --check-prefix=CHECK-HSW %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -mcpu knl -S -o - %s | FileCheck --check-prefix=CHECK-KNL %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -mcpu haswell -S -o - %s | FileCheck --check-prefix=CHECK-HSW %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -mcpu knl -S -o - %s | FileCheck --check-prefix=CHECK-KNL %s

define <8 x double> @loadwidth_insert_extract(ptr %ptr) {
    %b = getelementptr <2 x double>, ptr %ptr, i32 1
    %c = getelementptr <2 x double>, ptr %ptr, i32 2
    %d = getelementptr <2 x double>, ptr %ptr, i32 3
; CHECK-HSW: load <4 x double>
; CHECK-HSW: load <4 x double>
; CHECK-HSW-NOT: load
; CHECK-KNL: load <8 x double>
; CHECK-KNL-NOT: load
    %la = load <2 x double>, ptr %ptr
    %lb = load <2 x double>, ptr %b
    %lc = load <2 x double>, ptr %c
    %ld = load <2 x double>, ptr %d
    ; Scalarize everything - Explicitly not a shufflevector to test this code
    ; path in the LSV
    %v1 = extractelement <2 x double> %la, i32 0
    %v2 = extractelement <2 x double> %la, i32 1
    %v3 = extractelement <2 x double> %lb, i32 0
    %v4 = extractelement <2 x double> %lb, i32 1
    %v5 = extractelement <2 x double> %lc, i32 0
    %v6 = extractelement <2 x double> %lc, i32 1
    %v7 = extractelement <2 x double> %ld, i32 0
    %v8 = extractelement <2 x double> %ld, i32 1
    ; Make a vector again
    %i1 = insertelement <8 x double> undef, double %v1, i32 0
    %i2 = insertelement <8 x double> %i1, double %v2, i32 1
    %i3 = insertelement <8 x double> %i2, double %v3, i32 2
    %i4 = insertelement <8 x double> %i3, double %v4, i32 3
    %i5 = insertelement <8 x double> %i4, double %v5, i32 4
    %i6 = insertelement <8 x double> %i5, double %v6, i32 5
    %i7 = insertelement <8 x double> %i6, double %v7, i32 6
    %i8 = insertelement <8 x double> %i7, double %v8, i32 7
    ret <8 x double> %i8
}
