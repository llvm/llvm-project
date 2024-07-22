; RUN: not llc --mtriple=loongarch64 -mattr=+d < %s 2>&1 | FileCheck %s

declare ptr @llvm.returnaddress(i32 immarg)

define ptr @non_zero_returnaddress() nounwind {
; CHECK: return address can only be determined for the current frame
  %1 = call ptr @llvm.returnaddress(i32 1)
  ret ptr %1
}
