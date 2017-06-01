; RUN: not opt -verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_0(%X* %ptr) {
  %t = load %X, %X* %ptr
  ret void
; CHECK: loading unsized types is not allowed
; CHECK-NEXT:  %t = load %X, %X* %ptr
}

define void @f_1(%X %val, %X* %ptr) {
  store %X %val, %X* %ptr
  ret void
; CHECK: storing unsized types is not allowed
; CHECK-NEXT:  store %X %val, %X* %ptr
}

define void @f_2() {
  %t = alloca %X
  ret void
; CHECK: Cannot allocate unsized type
; CHECK-NEXT:  %t = alloca %X
}
