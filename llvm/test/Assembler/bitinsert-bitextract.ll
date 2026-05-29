; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK-LABEL: test_bitextract
; CHECK: bitextract i8, b32 %{{.*}}, i32 24
define i8 @test_bitextract(b32 %src) {
  %result = bitextract i8, b32 %src, i32 24
  ret i8 %result
}

; CHECK-LABEL: test_bitinsert
; CHECK: bitinsert b32 %{{.*}}, i8 %{{.*}}, i32 3
define b32 @test_bitinsert(b32 %base, i8 %val) {
  %result = bitinsert b32 %base, i8 %val, i32 3
  ret b32 %result
}
