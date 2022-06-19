; RUN: llc < %s -mtriple=aarch64 -mattr=+mte -aarch64-order-frame-objects=0 | FileCheck %s

declare void @use(i8* %p)
declare void @llvm.aarch64.settag(i8* %p, i64 %a)
declare void @llvm.aarch64.settag.zero(i8* %p, i64 %a)

define void @stg16_16() {
entry:
; CHECK-LABEL: stg16_16:
; CHECK: st2g sp, [sp], #32
; CHECK: ret
  %a = alloca i8, i32 16, align 16
  %b = alloca i8, i32 16, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 16)
  call void @llvm.aarch64.settag(i8* %b, i64 16)
  ret void
}

define i32 @stg16_16_16_16_ret() {
entry:
; CHECK-LABEL: stg16_16_16_16_ret:
; CHECK: mov  w0, wzr
; CHECK: st2g sp, [sp, #32]
; CHECK: st2g sp, [sp], #64
; CHECK: ret
  %a = alloca i8, i32 16, align 16
  %b = alloca i8, i32 16, align 16
  %c = alloca i8, i32 16, align 16
  %d = alloca i8, i32 16, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 16)
  call void @llvm.aarch64.settag(i8* %b, i64 16)
  call void @llvm.aarch64.settag(i8* %c, i64 16)
  call void @llvm.aarch64.settag(i8* %d, i64 16)
  ret i32 0
}

define void @stg16_16_16_16() {
entry:
; CHECK-LABEL: stg16_16_16_16:
; CHECK: st2g sp, [sp, #32]
; CHECK: st2g sp, [sp], #64
; CHECK: ret
  %a = alloca i8, i32 16, align 16
  %b = alloca i8, i32 16, align 16
  %c = alloca i8, i32 16, align 16
  %d = alloca i8, i32 16, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 16)
  call void @llvm.aarch64.settag(i8* %b, i64 16)
  call void @llvm.aarch64.settag(i8* %c, i64 16)
  call void @llvm.aarch64.settag(i8* %d, i64 16)
  ret void
}

define void @stg128_128_128_128() {
entry:
; CHECK-LABEL: stg128_128_128_128:
; CHECK: mov     x8, #512
; CHECK: st2g    sp, [sp], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
; CHECK: ret
  %a = alloca i8, i32 128, align 16
  %b = alloca i8, i32 128, align 16
  %c = alloca i8, i32 128, align 16
  %d = alloca i8, i32 128, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 128)
  call void @llvm.aarch64.settag(i8* %b, i64 128)
  call void @llvm.aarch64.settag(i8* %c, i64 128)
  call void @llvm.aarch64.settag(i8* %d, i64 128)
  ret void
}

define void @stg16_512_16() {
entry:
; CHECK-LABEL: stg16_512_16:
; CHECK: mov     x8, #544
; CHECK: st2g    sp, [sp], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
; CHECK: ret
  %a = alloca i8, i32 16, align 16
  %b = alloca i8, i32 512, align 16
  %c = alloca i8, i32 16, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 16)
  call void @llvm.aarch64.settag(i8* %b, i64 512)
  call void @llvm.aarch64.settag(i8* %c, i64 16)
  ret void
}

define void @stg512_512_512() {
entry:
; CHECK-LABEL: stg512_512_512:
; CHECK: mov     x8, #1536
; CHECK: st2g    sp, [sp], #32
; CHECK: sub     x8, x8, #32
; CHECK: cbnz    x8,
; CHECK: ret
  %a = alloca i8, i32 512, align 16
  %b = alloca i8, i32 512, align 16
  %c = alloca i8, i32 512, align 16
  call void @llvm.aarch64.settag(i8* %a, i64 512)
  call void @llvm.aarch64.settag(i8* %b, i64 512)
  call void @llvm.aarch64.settag(i8* %c, i64 512)
  ret void
}

define void @early(i1 %flag) {
entry:
; CHECK-LABEL: early:
; CHECK: tbz     w0, #0, [[LABEL:.LBB.*]]
; CHECK: st2g    sp, [sp, #
; CHECK: st2g    sp, [sp, #
; CHECK: st2g    sp, [sp, #
; CHECK: [[LABEL]]:
; CHECK: stg     sp, [sp, #
; CHECK: st2g    sp, [sp], #
; CHECK: ret
  %a = alloca i8, i32 48, align 16
  %b = alloca i8, i32 48, align 16
  %c = alloca i8, i32 48, align 16
  br i1 %flag, label %if.then, label %if.end

if.then:
  call void @llvm.aarch64.settag(i8* %a, i64 48)
  call void @llvm.aarch64.settag(i8* %b, i64 48)
  br label %if.end

if.end:
  call void @llvm.aarch64.settag(i8* %c, i64 48)
  ret void
}

define void @early_128_128(i1 %flag) {
entry:
; CHECK-LABEL: early_128_128:
; CHECK: tbz   w0, #0, [[LABEL:.LBB.*]]
; CHECK: add   x9, sp, #
; CHECK: mov   x8, #256
; CHECK: sub   x8, x8, #32
; CHECK: st2g  x9, [x9], #32
; CHECK: cbnz  x8,
; CHECK: [[LABEL]]:
; CHECK: stg     sp, [sp, #
; CHECK: st2g    sp, [sp], #
; CHECK: ret
  %a = alloca i8, i32 128, align 16
  %b = alloca i8, i32 128, align 16
  %c = alloca i8, i32 48, align 16
  br i1 %flag, label %if.then, label %if.end

if.then:
  call void @llvm.aarch64.settag(i8* %a, i64 128)
  call void @llvm.aarch64.settag(i8* %b, i64 128)
  br label %if.end

if.end:
  call void @llvm.aarch64.settag(i8* %c, i64 48)
  ret void
}

define void @early_512_512(i1 %flag) {
entry:
; CHECK-LABEL: early_512_512:
; CHECK: tbz   w0, #0, [[LABEL:.LBB.*]]
; CHECK: add   x9, sp, #
; CHECK: mov   x8, #1024
; CHECK: sub   x8, x8, #32
; CHECK: st2g  x9, [x9], #32
; CHECK: cbnz  x8,
; CHECK: [[LABEL]]:
; CHECK: stg     sp, [sp, #
; CHECK: st2g    sp, [sp], #
; CHECK: ret
  %a = alloca i8, i32 512, align 16
  %b = alloca i8, i32 512, align 16
  %c = alloca i8, i32 48, align 16
  br i1 %flag, label %if.then, label %if.end

if.then:
  call void @llvm.aarch64.settag(i8* %a, i64 512)
  call void @llvm.aarch64.settag(i8* %b, i64 512)
  br label %if.end

if.end:
  call void @llvm.aarch64.settag(i8* %c, i64 48)
  ret void
}

; Two loops of size 256; the second loop updates SP.
define void @stg128_128_gap_128_128() {
entry:
; CHECK-LABEL: stg128_128_gap_128_128:
; CHECK: mov     x9, sp
; CHECK: mov     x8, #256
; CHECK: sub     x8, x8, #32
; CHECK: st2g    x9, [x9], #32
; CHECK: cbnz    x8,
; CHECK: mov     x8, #256
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
