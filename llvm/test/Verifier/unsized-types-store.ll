; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_1(%X %val, ptr %ptr) {
  store %X %val, ptr %ptr
  ret void
; CHECK: storing unsized types is not allowed
; CHECK-NEXT:  store %X %val, ptr %ptr
}
