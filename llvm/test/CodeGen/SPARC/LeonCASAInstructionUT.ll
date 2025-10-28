; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon3 -mattr=+hasleoncasa -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=gr712rc -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon4 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=gr740 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=myriad2 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=myriad2.1 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=myriad2.2 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=myriad2.3 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2100 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2150 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2155 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2450 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2455 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2x5x -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2080 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2085 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2480 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2485 -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ma2x8x -o - | FileCheck %s

; CHECK-LABEL: casa_test
; CHECK-DAG:   mov 1, [[R0:%[a-z0-9]+]]
; CHECK-DAG:   mov %g0, [[R1:%[a-z0-9]+]]
; CHECK:       casa [{{%[a-z0-9]+}}] 10, [[R1]], [[R0]]
define void @casa_test(ptr %ptr) {
  %pair = cmpxchg ptr %ptr, i32 0, i32 1 monotonic monotonic
  %r = extractvalue { i32, i1 } %pair, 0
  %stored1  = icmp eq i32 %r, 0

  ret void
}
