; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; RUN: opt < %s -passes='module-inline' -S | FileCheck %s

define i32 @testByte(i8 %X) {
entry:
  %tmp = icmp ne i8 %X, 0
  %tmp.i = zext i1 %tmp to i32
  ret i32 %tmp.i
}

define i32 @main() {
; CHECK-LABEL: define i32 @main()
entry:
  %rslt = call i32 @testByte(i8 123)
; CHECK-NOT: call
  ret i32 %rslt
; CHECK: ret i32 1
}
