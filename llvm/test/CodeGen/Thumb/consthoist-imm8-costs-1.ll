; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none-unknown-musleabi"

@a = global i8 undef, align 4

; Check that store-merging generates a single str i32 rather than strb+strb+strh,
; i.e., -1 is not moved by constant-hoisting.
; CHECK: movs [[R1:r[0-9]+]], #255
; CHECK: lsls [[R2:r[0-9]+]], [[R1]], #16
; CHECK: str  [[R2]]
; CHECK: movs [[R3:r[0-9]+]], #255
; CHECK: lsls [[R4:r[0-9]+]], [[R3]], #16
; CHECK: str  [[R4]]
; CHECK-NOT: strh
; CHECK-NOT: strb

define void @ham() {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:
  store i8 0, ptr getelementptr inbounds (i8, ptr @a, i32 1), align 1
  store i8 0, ptr @a, align 4
  store i8 -1, ptr getelementptr inbounds (i8, ptr @a, i32 2), align 2
  store i8 0, ptr getelementptr inbounds (i8, ptr @a, i32 3), align 1
  br label %bb3

bb2:
  store i8 0, ptr getelementptr inbounds (i8, ptr @a, i32 9), align 1
  store i8 0, ptr getelementptr inbounds (i8, ptr @a, i32 8), align 4
  store i8 -1, ptr getelementptr inbounds (i8, ptr @a, i32 10), align 2
  store i8 0, ptr getelementptr inbounds (i8, ptr @a, i32 11), align 1
  br label %bb3

bb3:
  ret void
}
