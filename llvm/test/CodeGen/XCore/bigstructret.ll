; RUN: llc < %s -march=xcore | FileCheck %s

%0 = type { i32, i32, i32, i32 }
%1 = type { i32, i32, i32, i32, i32 }

; Structs of 4 words can be returned in registers
define internal fastcc %0 @ReturnBigStruct() nounwind readnone {
entry:
  %0 = insertvalue %0 zeroinitializer, i32 12, 0
  %1 = insertvalue %0 %0, i32 24, 1
  %2 = insertvalue %0 %1, i32 48, 2
  %3 = insertvalue %0 %2, i32 24601, 3
  ret %0 %3
}
; CHECK: ReturnBigStruct:
; CHECK: ldc r0, 12
; CHECK: ldc r1, 24
; CHECK: ldc r2, 48
; CHECK: ldc r3, 24601
; CHECK: retsp 0

; Structs bigger than 4 words are returned via a hidden hidden sret-parameter
define internal fastcc %1 @ReturnBigStruct2() nounwind readnone {
entry:
  %0 = insertvalue %1 zeroinitializer, i32 12, 0
  %1 = insertvalue %1 %0, i32 24, 1
  %2 = insertvalue %1 %1, i32 48, 2
  %3 = insertvalue %1 %2, i32 24601, 3
  %4 = insertvalue %1 %3, i32 4321, 4
  ret %1 %4
}
; CHECK: ReturnBigStruct2:
; CHECK: ldc r1, 4321
; CHECK: stw r1, r0[4]
; CHECK: ldc r1, 24601
; CHECK: stw r1, r0[3]
; CHECK: ldc r1, 48
; CHECK: stw r1, r0[2]
; CHECK: ldc r1, 24
; CHECK: stw r1, r0[1]
; CHECK: ldc r1, 12
; CHECK: stw r1, r0[0]
; CHECK: retsp 0
