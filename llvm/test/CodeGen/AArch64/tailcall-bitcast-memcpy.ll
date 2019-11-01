;RUN: llc %s -o - -verify-machineinstrs | FileCheck %s
target triple = "aarch64-arm-none-eabi"

;CHECK-LABEL: @wmemcpy
;CHECK: lsl
;CHECK-NOT: bl
;CHECK-NOT: mov
;CHECK-NOT: ldp
;CHECK-NEXT: b memcpy
define dso_local i32* @wmemcpy(i32* returned, i32* nocapture readonly, i64) local_unnamed_addr {
  %4 = bitcast i32* %0 to i8*
  %5 = bitcast i32* %1 to i8*
  %6 = shl i64 %2, 2
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 %6, i1 false)
  ret i32* %0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
