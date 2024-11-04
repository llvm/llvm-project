; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Check that intrinsics do not get automatically declared if they are used
; with different function types.

; CHECK: error: use of undefined value '@llvm.umax'
define void @test() {
  call i8 @llvm.umax(i8 0, i8 1)
  call i16 @llvm.umax(i16 0, i16 1)
  ret void
}
