; RUN: not opt -verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_1(%X %val, %X* %ptr) {
  store %X %val, %X* %ptr
  ret void
; CHECK: storing unsized types is not allowed
; CHECK-NEXT:  store %X %val, %X* %ptr
}
