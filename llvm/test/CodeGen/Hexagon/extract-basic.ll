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

define void @foo(%struct.structx_t* nocapture %px, %struct.structy_t* nocapture %py) nounwind {
entry:
  %0 = bitcast %struct.structy_t* %py to i32*
  %1 = load i32, i32* %0, align 4
  %bf.value = and i32 %1, 7
  %2 = bitcast %struct.structx_t* %px to i32*
  %3 = load i32, i32* %2, align 4
  %4 = and i32 %3, -8
  %5 = or i32 %4, %bf.value
  store i32 %5, i32* %2, align 4
  %6 = load i32, i32* %0, align 4
  %7 = lshr i32 %6, 4
  %bf.clear1 = shl nuw nsw i32 %7, 3
  %8 = and i32 %bf.clear1, 56
  %9 = and i32 %5, -1017
  %10 = or i32 %8, %9
  store i32 %10, i32* %2, align 4
  %11 = load i32, i32* %0, align 4
  %12 = lshr i32 %11, 7
  %bf.value4 = shl i32 %12, 10
  %13 = and i32 %bf.value4, 261120
  %14 = and i32 %10, -262081
  %15 = or i32 %14, %13
  store i32 %15, i32* %2, align 4
  %16 = load i32, i32* %0, align 4
  %17 = lshr i32 %16, 16
  %bf.clear5 = shl i32 %17, 18
  %18 = and i32 %bf.clear5, 66846720
  %19 = and i32 %15, -1073480641
  %20 = or i32 %19, %18
  store i32 %20, i32* %2, align 4
  %21 = load i32, i32* %0, align 4
  %22 = lshr i32 %21, 24
  %23 = shl i32 %22, 30
  %24 = and i32 %20, 67107903
  %25 = or i32 %24, %23
  store i32 %25, i32* %2, align 4
  ret void
}
