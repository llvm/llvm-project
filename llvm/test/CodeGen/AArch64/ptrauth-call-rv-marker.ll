; RUN: llc -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -verify-machineinstrs -global-isel -o - %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64e-apple-iphoneos"

declare i8* @foo0(i32)
declare i8* @foo1()

declare void @llvm.objc.release(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)

declare void @foo2(i8*)

declare void @foo(i64, i64, i64)

define void @rv_marker_ptrauth_blraa(i8* ()** %arg0, i64 %arg1) {
; CHECK-LABEL: rv_marker_ptrauth_blraa
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blraa [[ADDR]], x1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl objc_retainAutoreleasedReturnValue
;
entry:
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 0, i64 %arg1), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blraa_unsafeClaim(i8* ()** %arg0, i64 %arg1) {
; CHECK-LABEL: rv_marker_ptrauth_blraa_unsafeClaim
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blraa [[ADDR]], x1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl objc_unsafeClaimAutoreleasedReturnValue
;
entry:
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 0, i64 %arg1), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blraa_disc_imm16(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blraa_disc_imm16
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    mov x17, #45431
; CHECK-NEXT:    blrab [[ADDR]], x17
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 45431), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blraa_multiarg(i8* (i64, i64, i64)** %arg0, i64 %arg1, i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_ptrauth_blraa_multiarg
; CHECK:         mov  [[TMP:x[0-9]+]], x1
; CHECK-DAG:     ldr [[ADDR:x[0-9]+]]
; CHECK-DAG:     mov x0, x4
; CHECK-DAG:     mov x1, x3
; CHECK-NEXT:    blraa [[ADDR]], [[TMP]]
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
entry:
  %tmp0 = load i8* (i64, i64, i64)*, i8* (i64, i64, i64)** %arg0
  %call0 = call i8* %tmp0(i64 %c, i64 %b, i64 %a) [ "ptrauth"(i32 0, i64 %arg1), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blrab(i8* ()** %arg0, i64 %arg1) {
; CHECK-LABEL: rv_marker_ptrauth_blrab
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blrab [[ADDR]], x1
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 %arg1), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blrab_disc_imm16(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blrab_disc_imm16
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    mov x17, #256
; CHECK-NEXT:    blrab [[ADDR]], x17
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 256), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blraaz(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blraaz
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blraaz [[ADDR]]
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 0, i64 0), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blrabz(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blrabz
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blrabz [[ADDR]]
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call0 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 0), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}

define void @rv_marker_ptrauth_blrabz_multiarg(i8* (i64, i64, i64)** %arg0, i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_ptrauth_blrabz_multiarg
; CHECK:         mov  [[TMP:x[0-9]+]], x1
; CHECK-DAG:     ldr [[ADDR:x[0-9]+]], [
; CHECK-DAG:     mov x0, x3
; CHECK-DAG:     mov x1, x2
; CHECK-DAG:     mov x2, [[TMP]]
; CHECK-NEXT:    blrabz [[ADDR]]
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl objc_retainAutoreleasedReturnValue
;
  %tmp0 = load i8* (i64, i64, i64)*, i8* (i64, i64, i64)** %arg0
  %call0 = call i8* %tmp0(i64 %c, i64 %b, i64 %a) [ "ptrauth"(i32 1, i64 0), "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call0)
  tail call void @llvm.objc.release(i8* %call0)
  ret void
}
