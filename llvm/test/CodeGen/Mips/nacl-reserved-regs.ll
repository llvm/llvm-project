; RUN: llc -march=mipsel -O3 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-none-nacl-gnu -O3 < %s \
; RUN:  | FileCheck %s -check-prefix=CHECK-NACL

@var = external global i32

define void @f() {
  %val1 = load volatile i32, ptr @var
  %val2 = load volatile i32, ptr @var
  %val3 = load volatile i32, ptr @var
  %val4 = load volatile i32, ptr @var
  %val5 = load volatile i32, ptr @var
  %val6 = load volatile i32, ptr @var
  %val7 = load volatile i32, ptr @var
  %val8 = load volatile i32, ptr @var
  %val9 = load volatile i32, ptr @var
  %val10 = load volatile i32, ptr @var
  %val11 = load volatile i32, ptr @var
  %val12 = load volatile i32, ptr @var
  %val13 = load volatile i32, ptr @var
  %val14 = load volatile i32, ptr @var
  %val15 = load volatile i32, ptr @var
  %val16 = load volatile i32, ptr @var
  store volatile i32 %val1, ptr @var
  store volatile i32 %val2, ptr @var
  store volatile i32 %val3, ptr @var
  store volatile i32 %val4, ptr @var
  store volatile i32 %val5, ptr @var
  store volatile i32 %val6, ptr @var
  store volatile i32 %val7, ptr @var
  store volatile i32 %val8, ptr @var
  store volatile i32 %val9, ptr @var
  store volatile i32 %val10, ptr @var
  store volatile i32 %val11, ptr @var
  store volatile i32 %val12, ptr @var
  store volatile i32 %val13, ptr @var
  store volatile i32 %val14, ptr @var
  store volatile i32 %val15, ptr @var
  store volatile i32 %val16, ptr @var
  ret void

; Check that t6, t7 and t8 are used in non-NaCl code.
; CHECK:    lw  $14
; CHECK:    lw  $15
; CHECK:    lw  $24

; t6, t7 and t8 are reserved in NaCl.
; CHECK-NACL-NOT:    lw  $14
; CHECK-NACL-NOT:    lw  $15
; CHECK-NACL-NOT:    lw  $24
}
