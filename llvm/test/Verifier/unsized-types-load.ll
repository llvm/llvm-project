; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define void @f_0(ptr %ptr) {
  %t = load target("foo"), ptr %ptr
  ret void
; CHECK: loading unsized types is not allowed
; CHECK-NEXT:  %t = load target("foo"), ptr %ptr
}
