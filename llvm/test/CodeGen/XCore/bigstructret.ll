; RUN: llc < %s -mtriple=xcore | FileCheck %s

%0 = type { i32, i32, i32, i32 }
%1 = type { i32, i32, i32, i32, i32 }

; Structs of 4 words are returned in registers
define internal %0 @ReturnBigStruct() nounwind readnone {
entry:
  %0 = insertvalue %0 zeroinitializer, i32 12, 0
  %1 = insertvalue %0 %0, i32 24, 1
  %2 = insertvalue %0 %1, i32 48, 2
  %3 = insertvalue %0 %2, i32 24601, 3
  ret %0 %3
}
; CHECK-LABEL: ReturnBigStruct:
; CHECK: ldc r0, 12
; CHECK: ldc r1, 24
; CHECK: ldc r2, 48
; CHECK: ldc r3, 24601
; CHECK: retsp 0

; Structs of more than 4 words are partially returned in memory so long as the
; function is not variadic.
define { i32, i32, i32, i32, i32} @f(i32, i32, i32, i32, i32) nounwind readnone {
; CHECK-LABEL: f:
; CHECK: ldc [[REGISTER:r[0-9]+]], 5
; CHECK-NEXT: stw [[REGISTER]], sp[2]
; CHECK-NEXT: retsp 0
body:
  ret { i32, i32, i32, i32, i32} { i32 undef, i32 undef, i32 undef, i32 undef, i32 5}
}

@x = external global i32
@y = external global i32

; Check we call a function returning more than 4 words correctly.
define i32 @g() nounwind {
; CHECK-LABEL: g:
; CHECK: entsp 3
; CHECK: ldc [[REGISTER:r[0-9]+]], 0
; CHECK: stw [[REGISTER]], sp[1]
; CHECK: bl f
; CHECK-NEXT: ldw r0, sp[2]
; CHECK-NEXT: retsp 3
;
body:
  %0 = call { i32, i32, i32, i32, i32 } @f(i32 0, i32 0, i32 0, i32 0, i32 0)
  %1 = extractvalue { i32, i32, i32, i32, i32 } %0, 4
  ret i32 %1
}

; Variadic functions return structs bigger than 4 words via a hidden
; sret-parameter
define internal %1 @ReturnBigStruct2(i32 %dummy, ...) nounwind readnone {
entry:
  %0 = insertvalue %1 zeroinitializer, i32 12, 0
  %1 = insertvalue %1 %0, i32 24, 1
  %2 = insertvalue %1 %1, i32 48, 2
  %3 = insertvalue %1 %2, i32 24601, 3
  %4 = insertvalue %1 %3, i32 4321, 4
  ret %1 %4
}
; CHECK-LABEL: ReturnBigStruct2:
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
