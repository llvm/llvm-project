; RUN: llc < %s -mtriple=aarch64 -mattr=+mte -aarch64-order-frame-objects=1 | FileCheck %s

declare void @use(i8* %p)
declare void @llvm.aarch64.settag(i8* %p, i64 %a)
declare void @llvm.aarch64.settag.zero(i8* %p, i64 %a)

; Two loops of size 256; the second loop updates SP.
; After frame reordering, two loops can be merged into one.
define void @stg128_128_gap_128_128() {
entry:
; CHECK-LABEL: stg128_128_gap_128_128:
; CHECK: mov     x8, #512
; CHECK: st2g    sp, [sp], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
; CHECK: ret
  %a = alloca i8, i32 128, align 16
  %a2 = alloca i8, i32 128, align 16
  %b = alloca i8, i32 32, align 16
  %c = alloca i8, i32 128, align 16
  %c2 = alloca i8, i32 128, align 16
  call void @use(i8* %b)
  call void @llvm.aarch64.settag(i8* %a, i64 128)
  call void @llvm.aarch64.settag(i8* %a2, i64 128)
  call void @llvm.aarch64.settag(i8* %c, i64 128)
  call void @llvm.aarch64.settag(i8* %c2, i64 128)
  ret void
}

define void @stg2(i1 %flag) {
entry:
; CHECK-LABEL: stg2:
  %a = alloca i8, i32 160, align 16
  %a2 = alloca i8, i32 160, align 16
  %b = alloca i8, i32 32, align 16
  %c = alloca i8, i32 128, align 16
  %c2 = alloca i8, i32 128, align 16
  call void @use(i8* %b)
  br i1 %flag, label %if.then, label %if.else

if.then:
; CHECK: mov     x8, #320
; CHECK: st2g    x9, [x9], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
  call void @llvm.aarch64.settag(i8* %a, i64 160)
  call void @llvm.aarch64.settag(i8* %a2, i64 160)
  br label %if.end

if.else:
; CHECK: mov     x8, #256
; CHECK: st2g    x9, [x9], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
  call void @llvm.aarch64.settag(i8* %c, i64 128)
  call void @llvm.aarch64.settag(i8* %c2, i64 128)
  br label %if.end

if.end:
; CHECK: mov     x8, #576
; CHECK: st2g    sp, [sp], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
  call void @llvm.aarch64.settag(i8* %a, i64 160)
  call void @llvm.aarch64.settag(i8* %a2, i64 160)
  call void @llvm.aarch64.settag(i8* %c, i64 128)
  call void @llvm.aarch64.settag(i8* %c2, i64 128)

; CHECK: ret
  ret void
}
