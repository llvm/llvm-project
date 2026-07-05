; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @f_1(target("foo") %val, ptr %ptr) {
  store target("foo") %val, ptr %ptr
  ret void
; CHECK: storing unsized types is not allowed
; CHECK-NEXT:  store target("foo") %val, ptr %ptr
}
