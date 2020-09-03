; RUN: not opt -verify < %s 2>&1 | FileCheck %s

%X = type opaque

define void @f_2() {
  %t = alloca %X
  ret void
; CHECK: Cannot allocate unsized type
; CHECK-NEXT:  %t = alloca %X
}
