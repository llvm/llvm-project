; RUN: not opt -passes=verify -force-opaque-pointers -S < %s 2>&1 | FileCheck %s

declare i32 @llvm.umax.i32(i32, i32)

define void @intrinsic_signature_mismatch() {
; CHECK: Intrinsic called with incompatible signature
  call i32 @llvm.umax.i32(i32 0)
  ret void
}
