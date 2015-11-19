; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.eq $push[[NUM0:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.eq $push[[NUM1:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.and $push[[NUM2:[0-9]+]], $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ord_f64(double %x, double %y) {
  %a = fcmp ord double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.ne $push[[NUM0:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM2:[0-9]+]], $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @uno_f64(double %x, double %y) {
  %a = fcmp uno double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.eq $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oeq_f64(double %x, double %y) {
  %a = fcmp oeq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f64:
; CHECK: f64.ne $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @une_f64(double %x, double %y) {
  %a = fcmp une double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f64:
; CHECK: f64.lt $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @olt_f64(double %x, double %y) {
  %a = fcmp olt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f64:
; CHECK: f64.le $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ole_f64(double %x, double %y) {
  %a = fcmp ole double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f64:
; CHECK: f64.gt $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ogt_f64(double %x, double %y) {
  %a = fcmp ogt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f64:
; CHECK: f64.ge $push[[NUM:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oge_f64(double %x, double %y) {
  %a = fcmp oge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.eq $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @ueq_f64(double %x, double %y) {
  %a = fcmp ueq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result i32
; CHECK-NEXT: f64.ne $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.eq $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.eq $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.and $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.and $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]
define i32 @one_f64(double %x, double %y) {
  %a = fcmp one double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.lt $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @ult_f64(double %x, double %y) {
  %a = fcmp ult double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.le $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @ule_f64(double %x, double %y) {
  %a = fcmp ule double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.gt $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @ugt_f64(double %x, double %y) {
  %a = fcmp ugt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: f64.ge $push[[NUM0:[0-9]+]], $0, $1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]], $0, $0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM2:[0-9]+]], $1, $1{{$}}
; CHECK-NEXT: i32.or $push[[NUM3:[0-9]+]], $pop[[NUM1]], $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]], $pop[[NUM0]], $pop[[NUM3]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]{{$}}
define i32 @uge_f64(double %x, double %y) {
  %a = fcmp uge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
