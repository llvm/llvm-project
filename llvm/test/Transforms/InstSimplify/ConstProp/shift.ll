; RUN: opt < %s -instsimplify -S | FileCheck %s

; CHECK-LABEL: shift_undef_64
define void @shift_undef_64(i64* %p) {
  %r1 = lshr i64 -1, 4294967296 ; 2^32
  ; CHECK: store i64 poison
  store i64 %r1, i64* %p

  %r2 = ashr i64 -1, 4294967297 ; 2^32 + 1
  ; CHECK: store i64 poison
  store i64 %r2, i64* %p

  %r3 = shl i64 -1, 4294967298 ; 2^32 + 2
  ; CHECK: store i64 poison
  store i64 %r3, i64* %p

  ret void
}

; CHECK-LABEL: shift_undef_65
define void @shift_undef_65(i65* %p) {
  %r1 = lshr i65 2, 18446744073709551617
  ; CHECK: store i65 poison
  store i65 %r1, i65* %p

  %r2 = ashr i65 4, 18446744073709551617
  ; CHECK: store i65 poison
  store i65 %r2, i65* %p

  %r3 = shl i65 1, 18446744073709551617
  ; CHECK: store i65 poison
  store i65 %r3, i65* %p

  ret void
}

; CHECK-LABEL: shift_undef_256
define void @shift_undef_256(i256* %p) {
  %r1 = lshr i256 2, 18446744073709551617
  ; CHECK: store i256 poison
  store i256 %r1, i256* %p

  %r2 = ashr i256 4, 18446744073709551618
  ; CHECK: store i256 poison
  store i256 %r2, i256* %p

  %r3 = shl i256 1, 18446744073709551619
  ; CHECK: store i256 poison
  store i256 %r3, i256* %p

  ret void
}

; CHECK-LABEL: shift_undef_511
define void @shift_undef_511(i511* %p) {
  %r1 = lshr i511 -1, 1208925819614629174706276 ; 2^80 + 100
  ; CHECK: store i511 poison
  store i511 %r1, i511* %p

  %r2 = ashr i511 -2, 1208925819614629174706200
  ; CHECK: store i511 poison
  store i511 %r2, i511* %p

  %r3 = shl i511 -3, 1208925819614629174706180
  ; CHECK: store i511 poison
  store i511 %r3, i511* %p

  ret void
}
