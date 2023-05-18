; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_0(ptr %ptr) {
  %t = load %X, ptr %ptr
  ret void
; CHECK: loading unsized types is not allowed
; CHECK-NEXT:  %t = load %X, ptr %ptr
}
