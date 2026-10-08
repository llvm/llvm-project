; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Use of intrinsic without mangling suffix and invalid signature should
; be rejected.

; CHECK: error: intrinsic argument 1 type (matching overload type 0) expected i8, but got i16
define void @test() {
  call i8 @llvm.umax(i8 0, i16 1)
  ret void
}
