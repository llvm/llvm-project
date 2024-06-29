; RUN: opt -S -dxil-metadata-emit < %s | FileCheck %s --check-prefix=DXILMD

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

;
; cbuffer B
; {
;
;   struct B
;   {
;
;       double B0;                                    ; Offset:    0
;       float3 B1;                                    ; Offset:   16
;       float B2;                                     ; Offset:   28
;       double3 B3;                                   ; Offset:   32
;       half B4;                                      ; Offset:   56
;       double2 B5;                                   ; Offset:   64
;       float B6;                                     ; Offset:   80
;       half3 B7;                                     ; Offset:   84
;       half3 B8;                                     ; Offset:   90
;   
;   } B;                                              ; Offset:    0 Size:    96
;
; }
;


; Make sure the size is 96
; DXILMD:!{i32 0, ptr @B.cb., !"", i32 0, i32 1, i32 1, i32 96}

@B.cb. = external local_unnamed_addr constant { double, <3 x float>, float, <3 x double>, half, <2 x double>, float, <3 x half>, <3 x half> }


!hlsl.cbufs = !{!0}

!0 = !{ptr @B.cb., !"B.cb.ty", i32 13, i1 false, i32 1, i32 0}
