; RUN: llc -mtriple=arm64-apple-ios7.0 -mcpu=cyclone %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64_be-linux-gnu -mcpu=cyclone %s -o - | FileCheck --check-prefix=CHECK-BE %s

define i128 @test_128bitmul(i128 %lhs, i128 %rhs) {
; CHECK-LABEL: test_128bitmul:
; CHECK:       umulh [[HI:x[0-9]+]], x0, x2
; CHECK:       madd  [[TEMP1:x[0-9]+]], x0, x3, [[HI]]
; CHECK-DAG:   madd  x1, x1, x2, [[TEMP1]]
; CHECK-DAG:   mul   x0, x0, x2
; CHECK-NEXT:  ret

; CHECK-BE-LABEL: test_128bitmul:
; CHECK-BE:       umulh [[HI:x[0-9]+]], x1, x3
; CHECK-BE:       madd  [[TEMP1:x[0-9]+]], x1, x2, [[HI]]
; CHECK-BE-DAG:   madd  x0, x0, x3, [[TEMP1]]
; CHECK-BE-DAG:   mul   x1, x1, x3
; CHECK-BE-NEXT:  ret

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}

; The machine combiner should create madd instructions when
; optimizing for size because that's smaller than mul + add.

define i128 @test_128bitmul_optsize(i128 %lhs, i128 %rhs) optsize {
; CHECK-LABEL: test_128bitmul_optsize:
; CHECK:       umulh [[HI:x[0-9]+]], x0, x2
; CHECK-NEXT:  madd  [[TEMP1:x[0-9]+]], x0, x3, [[HI]]
; CHECK-DAG:   madd  x1, x1, x2, [[TEMP1]]
; CHECK-DAG:   mul   x0, x0, x2
; CHECK-NEXT:  ret

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}

define i128 @test_128bitmul_minsize(i128 %lhs, i128 %rhs) minsize {
; CHECK-LABEL: test_128bitmul_minsize:
; CHECK:       umulh [[HI:x[0-9]+]], x0, x2
; CHECK-NEXT:  madd  [[TEMP1:x[0-9]+]], x0, x3, [[HI]]
; CHECK-DAG:   madd  x1, x1, x2, [[TEMP1]]
; CHECK-DAG:   mul   x0, x0, x2
; CHECK-NEXT:  ret

  %prod = mul i128 %lhs, %rhs
  ret i128 %prod
}

