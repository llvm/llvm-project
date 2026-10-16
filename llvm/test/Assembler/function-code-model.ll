; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: define void @f1() code_model "tiny" {
define void @f1() code_model "tiny" {
  ret void
}

; CHECK: define void @f2() code_model "small" {
define void @f2() code_model "small" {
  ret void
}

; CHECK: define void @f3() code_model "kernel" {
define void @f3() code_model "kernel" {
  ret void
}

; CHECK: define void @f4() code_model "medium" {
define void @f4() code_model "medium" {
  ret void
}

; CHECK: define void @f5() code_model "large" {
define void @f5() code_model "large" {
  ret void
}
