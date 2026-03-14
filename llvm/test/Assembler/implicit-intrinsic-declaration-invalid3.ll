; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Use of unknown intrinsic without declaration should be rejected.

; CHECK: error: unknown intrinsic 'llvm.foobar'
define void @test() {
  call i8 @llvm.foobar(i8 0, i16 1)
  ret void
}
