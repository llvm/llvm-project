; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-ASLW
; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-ASRW
; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-LSRW
; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-ASLH
; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-ASRH
; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=CHECK-LSRH
;
; Make sure that the instructions with immediate operands are generated.
; CHECK-ASLW: vaslw({{.*}},#9)
; CHECK-ASRW: vasrw({{.*}},#8)
; CHECK-LSRW: vlsrw({{.*}},#7)
; CHECK-ASLH: vaslh({{.*}},#6)
; CHECK-ASRH: vasrh({{.*}},#5)
; CHECK-LSRH: vlsrh({{.*}},#4)

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i64 @foo(i64 %x) nounwind readnone {
entry:
  %0 = tail call i64 @llvm.hexagon.S2.asl.i.vw(i64 %x, i32 9)
  %1 = tail call i64 @llvm.hexagon.S2.asr.i.vw(i64 %x, i32 8)
  %2 = tail call i64 @llvm.hexagon.S2.lsr.i.vw(i64 %x, i32 7)
  %3 = tail call i64 @llvm.hexagon.S2.asl.i.vh(i64 %x, i32 6)
  %4 = tail call i64 @llvm.hexagon.S2.asr.i.vh(i64 %x, i32 5)
  %5 = tail call i64 @llvm.hexagon.S2.lsr.i.vh(i64 %x, i32 4)
  %add = add i64 %1, %0
  %add1 = add i64 %add, %2
  %add2 = add i64 %add1, %3
  %add3 = add i64 %add2, %4
  %add4 = add i64 %add3, %5
  ret i64 %add4
}

declare i64 @llvm.hexagon.S2.asl.i.vw(i64, i32) nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.vw(i64, i32) nounwind readnone
declare i64 @llvm.hexagon.S2.lsr.i.vw(i64, i32) nounwind readnone
declare i64 @llvm.hexagon.S2.asl.i.vh(i64, i32) nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.vh(i64, i32) nounwind readnone
declare i64 @llvm.hexagon.S2.lsr.i.vh(i64, i32) nounwind readnone

