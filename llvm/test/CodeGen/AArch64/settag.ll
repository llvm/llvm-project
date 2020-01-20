; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define void @stg1(i8* %p) {
entry:
; CHECK-LABEL: stg1:
; CHECK: stg x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 16)
  ret void
}

define void @stg2(i8* %p) {
entry:
; CHECK-LABEL: stg2:
; CHECK: st2g x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 32)
  ret void
}

define void @stg3(i8* %p) {
entry:
; CHECK-LABEL: stg3:
; CHECK: stg  x0, [x0, #32]
; CHECK: st2g x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 48)
  ret void
}

define void @stg4(i8* %p) {
entry:
; CHECK-LABEL: stg4:
; CHECK: st2g x0, [x0, #32]
; CHECK: st2g x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 64)
  ret void
}

define void @stg5(i8* %p) {
entry:
; CHECK-LABEL: stg5:
; CHECK: stg  x0, [x0, #64]
; CHECK: st2g x0, [x0, #32]
; CHECK: st2g x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 80)
  ret void
}

define void @stg16(i8* %p) {
entry:
; CHECK-LABEL: stg16:
; CHECK: mov  {{(w|x)}}[[R:[0-9]+]], #256
; CHECK: st2g x0, [x0], #32
; CHECK: sub  x[[R]], x[[R]], #32
; CHECK: cbnz x[[R]],
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 256)
  ret void
}

define void @stg17(i8* %p) {
entry:
; CHECK-LABEL: stg17:
; CHECK: stg x0, [x0], #16
; CHECK: mov  {{(w|x)}}[[R:[0-9]+]], #256
; CHECK: st2g x0, [x0], #32
; CHECK: sub  x[[R]], x[[R]], #32
; CHECK: cbnz x[[R]],
; CHECK: ret
  call void @llvm.aarch64.settag(i8* %p, i64 272)
  ret void
}

define void @stzg3(i8* %p) {
entry:
; CHECK-LABEL: stzg3:
; CHECK: stzg  x0, [x0, #32]
; CHECK: stz2g x0, [x0]
; CHECK: ret
  call void @llvm.aarch64.settag.zero(i8* %p, i64 48)
  ret void
}

define void @stzg17(i8* %p) {
entry:
; CHECK-LABEL: stzg17:
; CHECK: stzg x0, [x0], #16
; CHECK: mov  {{w|x}}[[R:[0-9]+]], #256
; CHECK: stz2g x0, [x0], #32
; CHECK: sub  x[[R]], x[[R]], #32
; CHECK: cbnz x[[R]],
; CHECK: ret
  call void @llvm.aarch64.settag.zero(i8* %p, i64 272)
  ret void
}

define void @stg_alloca1() {
entry:
; CHECK-LABEL: stg_alloca1:
; CHECK: stg sp, [sp]
; CHECK: ret
  %a = alloca i8, i32 16, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 16)
  ret void
}

define void @stg_alloca5() {
entry:
; CHECK-LABEL: stg_alloca5:
; CHECK:         st2g    sp, [sp, #32]
; CHECK-NEXT:    stg     sp, [sp, #64]
; CHECK-NEXT:    st2g    sp, [sp], #80
; CHECK-NEXT:    ret
  %a = alloca i8, i32 80, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 80)
  ret void
}

define void @stg_alloca17() {
entry:
; CHECK-LABEL: stg_alloca17:
; CHECK: mov  {{w|x}}[[R:[0-9]+]], #256
; CHECK: st2g sp, [sp], #32
; CHECK: sub  x[[R]], x[[R]], #32
; CHECK: cbnz x[[R]],
; CHECK: stg sp, [sp], #16
; CHECK: ret
  %a = alloca i8, i32 272, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 272)
  ret void
}

declare void @llvm.aarch64.settag(i8* %p, i64 %a)
declare void @llvm.aarch64.settag.zero(i8* %p, i64 %a)
