; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memset.i64(ptr captures(none), i8, i64)
declare void @llvm.memset.i64(ptr nocapture, i8, i64) nounwind

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memcpy.i64(ptr captures(none), i8, i64)
declare void @llvm.memcpy.i64(ptr nocapture, i8, i64) nounwind

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memmove.i64(ptr captures(none), i8, i64)
declare void @llvm.memmove.i64(ptr nocapture, i8, i64) nounwind
