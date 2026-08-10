; RUN: llvm-link -S %s %p/Inputs/pr27044.ll -o - | FileCheck %s

; CHECK: define i32 @f1() {
; CHECK: define i32 @g1() {
; CHECK: define void @f2() comdat($foo) {
; CHECK: define linkonce_odr i32 @g2() comdat($bar) {

define i32 @f1() {
  ret i32 0
}

define i32 @g1() {
  ret i32 0
}
