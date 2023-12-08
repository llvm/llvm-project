; RUN: llc -o - %s -asm-verbose=false -wasm-keep-registers -disable-wasm-fallthrough-return-opt -mattr=+simd128 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Test that stackified IMPLICIT_DEF instructions are converted into
; CONST_XXX instructions to provide an explicit push.

; CHECK-LABEL: implicit_def_i32:
; CHECK:      i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK:      i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK:      i32.const $push[[R:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i32 @implicit_def_i32() {
  br i1 undef, label %A, label %X

A:                                                ; preds = %0
  %d = icmp slt i1 0, 0
  br i1 %d, label %C, label %B

B:                                                ; preds = %A
  br label %C

C:                                                ; preds = %B, %A
  %h = phi i32 [ undef, %A ], [ 0, %B ]
  br label %X

X:                                                ; preds = %0, C
  %i = phi i32 [ 1, %0 ], [ %h, %C ]
  ret i32 %i
}

; CHECK-LABEL: implicit_def_i64:
; CHECK:      i64.const $push[[R:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define i64 @implicit_def_i64() {
  br i1 undef, label %A, label %X

A:                                                ; preds = %0
  %d = icmp slt i1 0, 0
  br i1 %d, label %C, label %B

B:                                                ; preds = %A
  br label %C

C:                                                ; preds = %B, %A
  %h = phi i64 [ undef, %A ], [ 0, %B ]
  br label %X

X:                                                ; preds = %0, C
  %i = phi i64 [ 1, %0 ], [ %h, %C ]
  ret i64 %i
}

; CHECK-LABEL: implicit_def_f32:
; CHECK:      f32.const $push[[R:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define float @implicit_def_f32() {
  br i1 undef, label %A, label %X

A:                                                ; preds = %0
  %d = icmp slt i1 0, 0
  br i1 %d, label %C, label %B

B:                                                ; preds = %A
  br label %C

C:                                                ; preds = %B, %A
  %h = phi float [ undef, %A ], [ 0.0, %B ]
  br label %X

X:                                                ; preds = %0, C
  %i = phi float [ 1.0, %0 ], [ %h, %C ]
  ret float %i
}

; CHECK-LABEL: implicit_def_f64:
; CHECK:      f64.const $push[[R:[0-9]+]]=, 0x0p0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define double @implicit_def_f64() {
  br i1 undef, label %A, label %X

A:                                                ; preds = %0
  %d = icmp slt i1 0, 0
  br i1 %d, label %C, label %B

B:                                                ; preds = %A
  br label %C

C:                                                ; preds = %B, %A
  %h = phi double [ undef, %A ], [ 0.0, %B ]
  br label %X

X:                                                ; preds = %0, C
  %i = phi double [ 1.0, %0 ], [ %h, %C ]
  ret double %i
}

; CHECK-LABEL: implicit_def_v4i32:
; CHECK:      v128.const $push[[R:[0-9]+]]=, 0, 0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @implicit_def_v4i32() {
  br i1 undef, label %A, label %X

A:                                                ; preds = %0
  %d = icmp slt i1 0, 0
  br i1 %d, label %C, label %B

B:                                                ; preds = %A
  br label %C

C:                                                ; preds = %B, %A
  %h = phi <4 x i32> [ undef, %A ], [ <i32 0, i32 0, i32 0, i32 0>, %B ]
  br label %X

X:                                                ; preds = %0, C
  %i = phi <4 x i32> [ <i32 1, i32 1, i32 1, i32 1>, %0 ], [ %h, %C ]
  ret <4 x i32> %i
}
