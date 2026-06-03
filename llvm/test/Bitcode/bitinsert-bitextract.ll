; RUN: llvm-as %s -o - -f | llvm-dis | FileCheck %s
; RUN: llvm-as %s -o - -f | verify-uselistorder

define b32 @test_bitinsert(b32 %x, i8 %y) {
; CHECK: %r = bitinsert b32 %x, i8 %y, i32 3
  %r = bitinsert b32 %x, i8 %y, i32 3
  ret b32 %r
}

define i8 @test_bitextract(b32 %src) {
; CHECK: %r = bitextract i8, b32 %src, i32 24
  %r = bitextract i8, b32 %src, i32 24
  ret i8 %r
}