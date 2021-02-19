; RUN: llc -o - %s | FileCheck --check-prefix=SELDAG --check-prefix=CHECK %s
; RUN: llc -global-isel -o - %s | FileCheck --check-prefix=GISEL --check-prefix=CHECK %s

; TODO: support marker generation with GlobalISel
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64e-apple-iphoneos"

declare i8* @foo0(i32)
declare i8* @foo1()

declare void @llvm.objc.release(i8*)

declare void @foo2(i8*)

declare void @foo(i64, i64, i64)

define void @rv_marker_ptrauth_blraa(i8* ()** %arg0, i64 %arg1) {
; CHECK-LABEL: rv_marker_ptrauth_blraa
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:   blraa [[ADDR]], x1
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
entry:
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 0, i64 %arg1), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blraa_disc_imm16(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blraa_disc_imm16
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:   mov w[[DISC:[0-9]+]], #45431
; SELDAG-NEXT:   blrab [[ADDR]], x[[DISC]]
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 45431), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blraa_multiarg(i8* (i64, i64, i64)** %arg0, i64 %arg1, i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_ptrauth_blraa_multiarg
; SELDAG:         mov  [[TMP:x[0-9]+]], x1
; CHECK:          ldr [[ADDR:x[0-9]+]]
; SELDAG-NEXT:    mov x0, x4
; SELDAG-NEXT:    mov x1, x3
; SELDAG-NEXT:    blraa [[ADDR]], [[TMP]]
; SELDAG-NEXT:    mov x29, x29
; GISEL:          blr [[ADDR]]
; GISEL-NOT:      mov x29, x29
;
entry:
  %tmp0 = load i8* (i64, i64, i64)*, i8* (i64, i64, i64)** %arg0
  %call1 = call i8* %tmp0(i64 %c, i64 %b, i64 %a) [ "ptrauth"(i32 0, i64 %arg1), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blrab(i8* ()** %arg0, i64 %arg1) {
; CHECK-LABEL: rv_marker_ptrauth_blrab
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:    blrab [[ADDR]], x1
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 %arg1), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blrab_disc_imm16(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blrab_disc_imm16
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:    mov w[[DISC:[0-9]+]], #256
; SELDAG-NEXT:   blrab [[ADDR]], x[[DISC]]
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 256), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blraaz(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blraaz
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:   blraaz [[ADDR]]
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 0, i64 0), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blrabz(i8* ()** %arg0) {
; CHECK-LABEL: rv_marker_ptrauth_blrabz
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:   blrabz [[ADDR]]
; SELDAG-NEXT:   mov x29, x29
; GISEL-NEXT:    blr [[ADDR]]
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* ()*, i8* ()** %arg0
  %call1 = call i8* %tmp0() [ "ptrauth"(i32 1, i64 0), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}

define void @rv_marker_ptrauth_blrabz_multiarg(i8* (i64, i64, i64)** %arg0, i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_ptrauth_blrabz_multiarg
; CHECK:         mov  [[TMP:x[0-9]+]], x1
; SELDAG-NEXT:   ldr [[ADDR:x[0-9]+]], [
; SELDAG-NEXT:   mov x0, x3
; SELDAG-NEXT:   mov x1, x2
; SELDAG-NEXT:   mov x2, [[TMP]]
; SELDAG-NEXT:   blrabz [[ADDR]]
; SELDAG-NEXT:   mov x29, x29
; GISEL:         blrabz
; GISEL-NOT:     mov x29, x29
;
  %tmp0 = load i8* (i64, i64, i64)*, i8* (i64, i64, i64)** %arg0
  %call1 = call i8* %tmp0(i64 %c, i64 %b, i64 %a) [ "ptrauth"(i32 1, i64 0), "clang.arc.attachedcall"(i64 0) ]
  tail call void @foo2(i8* %call1)
  tail call void @llvm.objc.release(i8* %call1)
  ret void
}
