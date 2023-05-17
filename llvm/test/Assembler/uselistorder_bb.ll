; RUN: llvm-as < %s -disable-output 2>&1 | FileCheck %s -allow-empty
; CHECK-NOT: error
; CHECK-NOT: warning
; RUN: verify-uselistorder < %s

@ba1 = constant ptr blockaddress (@bafunc1, %bb)
@ba2 = constant ptr getelementptr (i8, ptr blockaddress (@bafunc2, %bb), i61 0)
@ba3 = constant ptr getelementptr (i8, ptr blockaddress (@bafunc2, %bb), i61 0)

define ptr @babefore() {
  ret ptr getelementptr (i8, ptr blockaddress (@bafunc2, %bb), i61 0)
bb1:
  ret ptr blockaddress (@bafunc1, %bb)
bb2:
  ret ptr blockaddress (@bafunc3, %bb)
}
define void @bafunc1() {
  br label %bb
bb:
  unreachable
}
define void @bafunc2() {
  br label %bb
bb:
  unreachable
}
define void @bafunc3() {
  br label %bb
bb:
  unreachable
}
define ptr @baafter() {
  ret ptr blockaddress (@bafunc2, %bb)
bb1:
  ret ptr blockaddress (@bafunc1, %bb)
bb2:
  ret ptr blockaddress (@bafunc3, %bb)
}

uselistorder_bb @bafunc1, %bb, { 1, 0 }
uselistorder_bb @bafunc2, %bb, { 1, 0 }
uselistorder_bb @bafunc3, %bb, { 1, 0 }
