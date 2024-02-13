; RUN: opt -S -dxil-metadata-emit < %s | FileCheck %s --check-prefix=DXILMD

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; cbuffer B
; {
;
;   struct B
;   {
;
;       double B0[2];                                 ; Offset:    0
;       float3 B1[3];                                 ; Offset:   32
;       float B2;                                     ; Offset:   76
;       double B3[3];                                 ; Offset:   80
;       half B4;                                      ; Offset:  120
;       double2 B5[1];                                ; Offset:  128
;       float B6;                                     ; Offset:  144
;       half3 B7[2];                                  ; Offset:  160
;       half3 B8;                                     ; Offset:  182
;   
;   } B;                                              ; Offset:    0 Size:   188
;
; }
;
; cbuffer B
; {
;
;   struct B.0
;   {
;
;       double3 B9[3];                                ; Offset:    0
;       half3 B10;                                    ; Offset:   88
;   
;   } B;                                              ; Offset:    0 Size:    94
;
; }


; Make sure the size is 188.
; DXILMD:!{i32 0, ptr @B.cb., !"", i32 0, i32 1, i32 1, i32 188}
; Make sure the size is 94.
; DXILMD:!{i32 1, ptr @B.cb..1, !"", i32 0, i32 2, i32 1, i32 94}

@B.cb. = external local_unnamed_addr constant { [2 x double], [3 x <3 x float>], float, [3 x double], half, [1 x <2 x double>], float, [2 x <3 x half>], <3 x half> }
@B.cb..1 = external local_unnamed_addr constant { [3 x <3 x double>], <3 x half> }

!hlsl.cbufs = !{!0, !1}

!0 = !{ptr @B.cb., !"B.cb.ty", i32 13, i1 false, i32 1, i32 0}
!1 = !{ptr @B.cb..1, !"B.cb.ty", i32 13, i1 false, i32 2, i32 0}
