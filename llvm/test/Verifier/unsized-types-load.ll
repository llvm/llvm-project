; RUN: not opt -verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_0(%X* %ptr) {
  %t = load %X, %X* %ptr
  ret void
; CHECK: loading unsized types is not allowed
; CHECK-NEXT:  %t = load %X, %X* %ptr
}
