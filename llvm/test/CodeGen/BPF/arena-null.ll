; RUN: opt -S -mtriple=bpfel -mattr=+arena -passes='default<O2>' < %s | FileCheck %s
; RUN: opt -S -mtriple=bpfel -passes='default<O2>' < %s | FileCheck --check-prefix=OFF %s

; An access to arena through NULL reads 0. It's not a reason to drop code.

; CHECK-LABEL: define i32 @f(
; CHECK:       load i32, ptr null
; OFF-LABEL:   define i32 @f(
; OFF-NOT:     load
define i32 @f() {
  %v = load i32, ptr null
  ret i32 %v
}
