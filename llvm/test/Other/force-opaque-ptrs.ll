; RUN: llvm-as --force-opaque-pointers < %s | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis --force-opaque-pointers | FileCheck %s
; RUN: opt --force-opaque-pointers < %s -S | FileCheck %s

; CHECK: define void @f(ptr %p)
; CHECK:   ret void
define void @f(i32* %p) {
  ret void
}
