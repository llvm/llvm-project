; RUN: llc -O2 -march=hexagon < %s | FileCheck %s

; CHECK-DAG: extractu(r{{[0-9]*}},#3,#4)
; CHECK-DAG: extractu(r{{[0-9]*}},#8,#7)
; CHECK-DAG: extractu(r{{[0-9]*}},#8,#16)

; C source:
; typedef struct {
;   unsigned x1:3;
;   unsigned x2:7;
;   unsigned x3:8;
;   unsigned x4:12;
;   unsigned x5:2;
; } structx_t;
;
; typedef struct {
;   unsigned y1:4;
;   unsigned y2:3;
;   unsigned y3:9;
;   unsigned y4:8;
;   unsigned y5:8;
; } structy_t;
;
; void foo(structx_t *px, structy_t *py) {
;   px->x1 = py->y1;
;   px->x2 = py->y2;
;   px->x3 = py->y3;
;   px->x4 = py->y4;
;   px->x5 = py->y5;
; }

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.structx_t = type { i8, i8, i8, i8 }
%struct.structy_t = type { i8, i8, i8, i8 }

define void @foo(ptr nocapture %px, ptr nocapture %py) nounwind {
entry:
  %0 = load i32, ptr %py, align 4
  %bf.value = and i32 %0, 7
  %1 = load i32, ptr %px, align 4
  %2 = and i32 %1, -8
  %3 = or i32 %2, %bf.value
  store i32 %3, ptr %px, align 4
  %4 = load i32, ptr %py, align 4
  %5 = lshr i32 %4, 4
  %bf.clear1 = shl nuw nsw i32 %5, 3
  %6 = and i32 %bf.clear1, 56
  %7 = and i32 %3, -1017
  %8 = or i32 %6, %7
  store i32 %8, ptr %px, align 4
  %9 = load i32, ptr %py, align 4
  %10 = lshr i32 %9, 7
  %bf.value4 = shl i32 %10, 10
  %11 = and i32 %bf.value4, 261120
  %12 = and i32 %8, -262081
  %13 = or i32 %12, %11
  store i32 %13, ptr %px, align 4
  %14 = load i32, ptr %py, align 4
  %15 = lshr i32 %14, 16
  %bf.clear5 = shl i32 %15, 18
  %16 = and i32 %bf.clear5, 66846720
  %17 = and i32 %13, -1073480641
  %18 = or i32 %17, %16
  store i32 %18, ptr %px, align 4
  %19 = load i32, ptr %py, align 4
  %20 = lshr i32 %19, 24
  %21 = shl i32 %20, 30
  %22 = and i32 %18, 67107903
  %23 = or i32 %22, %21
  store i32 %23, ptr %px, align 4
  ret void
}
