; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

; @llvm.aarch64.cls must be directly translated into the 'cls' instruction

; CHECK-LABEL: cls
; CHECK: cls [[REG:w[0-9]+]], [[REG]]
define i32 @cls(i32 %t) {
  %cls.i = call i32 @llvm.aarch64.cls(i32 %t)
  ret i32 %cls.i
}

; CHECK-LABEL: cls64
; CHECK: cls [[REG:x[0-9]+]], [[REG]]
define i32 @cls64(i64 %t) {
  %cls.i = call i32 @llvm.aarch64.cls64(i64 %t)
  ret i32 %cls.i
}

declare i32 @llvm.aarch64.cls(i32) nounwind
declare i32 @llvm.aarch64.cls64(i64) nounwind
