; RUN: not llvm-as < %s 2>&1 | FileCheck %s

declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.my.custom.intrinsic()

; CHECK: Invalid user of intrinsic instruction!
@g1 = global ptr @llvm.umax.i32
; CHECK: Invalid user of intrinsic instruction!
@g2 = global ptr @llvm.my.custom.intrinsic
