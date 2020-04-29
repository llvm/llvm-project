; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: danny
; CHECK: memw_locked
define i8 @danny(i8* %a0) unnamed_addr #0 {
start:
  %v0 = cmpxchg i8* %a0, i8 0, i8 1 seq_cst seq_cst
  %v1 = extractvalue { i8, i1 } %v0, 0
  ret i8 %v1
}

; CHECK-LABEL: sammy
; CHECK: memw_locked
define i16 @sammy(i16* %a0) unnamed_addr #0 {
start:
  %v0 = cmpxchg i16* %a0, i16 0, i16 1 seq_cst seq_cst
  %v1 = extractvalue { i16, i1 } %v0, 0
  ret i16 %v1
}

; CHECK-LABEL: kirby
; CHECK: memw_locked
define i32 @kirby(i32* %a0) unnamed_addr #0 {
start:
  %v0 = cmpxchg i32* %a0, i32 0, i32 1 seq_cst seq_cst
  %v1 = extractvalue { i32, i1 } %v0, 0
  ret i32 %v1
}
