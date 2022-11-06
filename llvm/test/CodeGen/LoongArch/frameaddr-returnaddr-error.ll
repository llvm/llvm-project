; RUN: not llc --mtriple=loongarch64 --disable-verify < %s 2>&1 | FileCheck %s

declare ptr @llvm.frameaddress(i32)
declare ptr @llvm.returnaddress(i32)

define ptr @non_const_depth_frameaddress(i32 %x) nounwind {
; CHECK: argument to '__builtin_frame_address' must be a constant integer
  %1 = call ptr @llvm.frameaddress(i32 %x)
  ret ptr %1
}


define ptr @non_const_depth_returnaddress(i32 %x) nounwind {
; CHECK: argument to '__builtin_return_address' must be a constant integer
  %1 = call ptr @llvm.returnaddress(i32 %x)
  ret ptr %1
}

define ptr @non_zero_frameaddress() nounwind {
; CHECK: frame address can only be determined for the current frame
  %1 = call ptr @llvm.frameaddress(i32 1)
  ret ptr %1
}


define ptr @non_zero_returnaddress() nounwind {
; CHECK: return address can only be determined for the current frame
  %1 = call ptr @llvm.returnaddress(i32 1)
  ret ptr %1
}

