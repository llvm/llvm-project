; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare ptr @llvm.frameaddress(i32 immarg)
declare ptr @llvm.returnaddress(i32 immarg)

define ptr @non_const_depth_frameaddress(i32 %x) nounwind {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %x
  ; CHECK-NEXT: %1 = call ptr @llvm.frameaddress.p0(i32 %x)
  %1 = call ptr @llvm.frameaddress(i32 %x)
  ret ptr %1
}

define ptr @non_const_depth_returnaddress(i32 %x) nounwind {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %x
  ; CHECK-NEXT: %1 = call ptr @llvm.returnaddress(i32 %x)
  %1 = call ptr @llvm.returnaddress(i32 %x)
  ret ptr %1
}
