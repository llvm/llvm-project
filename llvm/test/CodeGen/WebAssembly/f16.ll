; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -fast-isel | FileCheck %s

; Test that f16 is expanded.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: demote.f32:
; CHECK-NEXT: .functype demote.f32 (f32) -> (f32){{$}}
; CHECK-NEXT: local.get	$push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: call	$push[[L1:[0-9]+]]=, __truncsfhf2, $pop[[L0]]{{$}}
; CHECK-NEXT: call	$push[[L2:[0-9]+]]=, __extendhfsf2, $pop[[L1]]{{$}}
; CHECK-NEXT: return  	$pop[[L2]]{{$}}
define half @demote.f32(float %f) {
    %t = fptrunc float %f to half
    ret half %t
}

; CHECK-LABEL: promote.f32:
; CHECK-NEXT: .functype promote.f32 (f32) -> (f32){{$}}
; CHECK-NEXT: local.get	$push0=, 0{{$}}
; CHECK-NEXT: return  	$pop0{{$}}
define float @promote.f32(half %f) {
    %t = fpext half %f to float
    ret float %t
}

; CHECK-LABEL: demote.f64:
; CHECK-NEXT: .functype demote.f64 (f64) -> (f32){{$}}
; CHECK-NEXT: local.get	$push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: call	$push[[L1:[0-9]+]]=, __truncdfhf2, $pop[[L0]]{{$}}
; CHECK-NEXT: call	$push[[L2:[0-9]+]]=, __extendhfsf2, $pop[[L1]]{{$}}
; CHECK-NEXT: return  	$pop[[L2]]{{$}}
define half @demote.f64(double %f) {
    %t = fptrunc double %f to half
    ret half %t
}

; CHECK-LABEL: promote.f64:
; CHECK-NEXT: .functype promote.f64 (f32) -> (f64){{$}}
; CHECK-NEXT: local.get	$push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f64.promote_f32 $push[[L1:[0-9]+]]=, $pop[[L0]]{{$}}
; CHECK-NEXT: return  	$pop[[L1]]{{$}}
define double @promote.f64(half %f) {
    %t = fpext half %f to double
    ret double %t
}

; CHECK-LABEL: demote.f128:
; CHECK-NEXT: .functype demote.f128 (i64, i64) -> (f32){{$}}
; CHECK-NEXT: local.get	$push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: call	$push[[L2:[0-9]+]]=, __trunctfhf2, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: call	$push[[L3:[0-9]+]]=, __extendhfsf2, $pop[[L2]]{{$}}
; CHECK-NEXT: return  	$pop[[L3]]{{$}}
define half @demote.f128(fp128 %f) {
    %t = fptrunc fp128 %f to half
    ret half %t
}

; CHECK-LABEL: promote.f128:
; CHECK-NEXT: .functype promote.f128 (i32, f32) -> (){{$}}
; CHECK: call __extendsftf2
; CHECK: i64.store
; CHECK: i64.store
define fp128 @promote.f128(half %f) {
    %t = fpext half %f to fp128
    ret fp128 %t
}
