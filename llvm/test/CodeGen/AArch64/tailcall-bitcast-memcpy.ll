;RUN: llc %s -o - -verify-machineinstrs | FileCheck %s
target triple = "aarch64"

;CHECK-LABEL: @wmemcpy
;CHECK: lsl
;CHECK-NOT: bl
;CHECK-NOT: mov
;CHECK-NOT: ldp
;CHECK-NEXT: b memcpy
define dso_local ptr @wmemcpy(ptr returned, ptr nocapture readonly, i64) local_unnamed_addr {
  %4 = shl i64 %2, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 4 %1, i64 %4, i1 false)
  ret ptr %0
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
