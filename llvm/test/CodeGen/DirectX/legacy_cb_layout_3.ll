; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s --check-prefix=DXILMD

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; cbuffer D
; {
;
;   struct D
;   {
;
;       int D0;                                       ; Offset:    0
;       struct struct.B
;       {
;
;           double B0;                                ; Offset:   16
;           float3 B1;                                ; Offset:   32
;           float B2;                                 ; Offset:   44
;           double3 B3;                               ; Offset:   48
;           half B4;                                  ; Offset:   72
;           double2 B5;                               ; Offset:   80
;           float B6;                                 ; Offset:   96
;           half3 B7;                                 ; Offset:  100
;           half3 B8;                                 ; Offset:  106
;       
;       } D1;                                         ; Offset:   16
;
;       half D2;                                      ; Offset:  112
;       struct struct.C
;       {
;
;           struct struct.A
;           {
;
;               float A0;                             ; Offset:  128
;               double A1;                            ; Offset:  136
;               float A2;                             ; Offset:  144
;               half A3;                              ; Offset:  148
;               int16_t A4;                           ; Offset:  150
;               int64_t A5;                           ; Offset:  152
;               int A6;                               ; Offset:  160
;           
;           } C0;                                     ; Offset:  128
;
;           float C1[1];                              ; Offset:  176
;           struct struct.B
;           {
;
;               double B0;                            ; Offset:  192
;               float3 B1;                            ; Offset:  208
;               float B2;                             ; Offset:  220
;               double3 B3;                           ; Offset:  224
;               half B4;                              ; Offset:  248
;               double2 B5;                           ; Offset:  256
;               float B6;                             ; Offset:  272
;               half3 B7;                             ; Offset:  276
;               half3 B8;                             ; Offset:  282
;           
;           } C2[2];;                                 ; Offset:  192
;
;           half C3;                                  ; Offset:  384
;       
;       } D3;                                         ; Offset:  128
;
;       double D4;                                    ; Offset:  392
;   
;   } D;                                              ; Offset:    0 Size:   400


; Make sure the size is 400
; DXILMD:!{i32 0, ptr @D.cb., !"", i32 0, i32 1, i32 1, i32 400}


%struct.B = type <{ double, <3 x float>, float, <3 x double>, half, <2 x double>, float, <3 x half>, <3 x half> }>
%struct.C = type <{ %struct.A, [1 x float], [2 x %struct.B], half }>
%struct.A = type <{ float, double, float, half, i16, i64, i32 }>

@D.cb. = external local_unnamed_addr constant { i32, %struct.B, half, %struct.C, double }

!hlsl.cbufs = !{!0}
!0 = !{ptr @D.cb., !"D.cb.ty", i32 13, i1 false, i32 1, i32 0}
