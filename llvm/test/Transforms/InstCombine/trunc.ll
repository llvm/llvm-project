; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Instcombine should be able to eliminate all of these ext casts.

declare void @use(i32)

define i64 @test1(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 15
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK-LABEL: @test1(
; CHECK-NOT: ext
; CHECK: ret
}
define i64 @test2(i64 %a) {
  %b = trunc i64 %a to i32
  %c = shl i32 %b, 4
  %q = ashr i32 %c, 4
  %d = sext i32 %q to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK-LABEL: @test2(
; CHECK: shl i64 %a, 36
; CHECK: %d = ashr exact i64 {{.*}}, 36
; CHECK: ret i64 %d
}
define i64 @test3(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK-LABEL: @test3(
; CHECK-NOT: ext
; CHECK: ret
}
define i64 @test4(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %x = xor i32 %c, 8
  %d = zext i32 %x to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK-LABEL: @test4(
; CHECK: = and i64 %a, 8
; CHECK: = xor i64 {{.*}}, 8
; CHECK-NOT: ext
; CHECK: ret
}

define i32 @test5(i32 %A) {
  %B = zext i32 %A to i128
  %C = lshr i128 %B, 16
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK-LABEL: @test5(
; CHECK: %C = lshr i32 %A, 16
; CHECK: ret i32 %C
}

define i32 @test6(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK-LABEL: @test6(
; CHECK: %C = lshr i64 %A, 32
; CHECK: %D = trunc i64 %C to i32
; CHECK: ret i32 %D
}

define i92 @test7(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i92
  ret i92 %D
; CHECK-LABEL: @test7(
; CHECK: %B = zext i64 %A to i92
; CHECK: %C = lshr i92 %B, 32
; CHECK: ret i92 %C
}

define i64 @test8(i32 %A, i32 %B) {
  %tmp38 = zext i32 %A to i128
  %tmp32 = zext i32 %B to i128
  %tmp33 = shl i128 %tmp32, 32
  %ins35 = or i128 %tmp33, %tmp38
  %tmp42 = trunc i128 %ins35 to i64
  ret i64 %tmp42
; CHECK-LABEL: @test8(
; CHECK:   %tmp38 = zext i32 %A to i64
; CHECK:   %tmp32 = zext i32 %B to i64
; CHECK:   %tmp33 = shl nuw i64 %tmp32, 32
; CHECK:   %ins35 = or i64 %tmp33, %tmp38
; CHECK:   ret i64 %ins35
}

define i8 @test9(i32 %X) {
  %Y = and i32 %X, 42
  %Z = trunc i32 %Y to i8
  ret i8 %Z
; CHECK-LABEL: @test9(
; CHECK: trunc
; CHECK: and
; CHECK: ret
}

; rdar://8808586
define i8 @test10(i32 %X) {
  %Y = trunc i32 %X to i8
  %Z = and i8 %Y, 42
  ret i8 %Z
; CHECK-LABEL: @test10(
; CHECK: trunc
; CHECK: and
; CHECK: ret
}

; PR25543
; https://llvm.org/bugs/show_bug.cgi?id=25543
; This is an extractelement.

define i32 @trunc_bitcast1(<4 x i32> %v) {
  %bc = bitcast <4 x i32> %v to i128
  %shr = lshr i128 %bc, 32
  %ext = trunc i128 %shr to i32
  ret i32 %ext

; CHECK-LABEL: @trunc_bitcast1(
; CHECK-NEXT:  %ext = extractelement <4 x i32> %v, i32 1
; CHECK-NEXT:  ret i32 %ext
}

; A bitcast may still be required.

define i32 @trunc_bitcast2(<2 x i64> %v) {
  %bc = bitcast <2 x i64> %v to i128
  %shr = lshr i128 %bc, 64
  %ext = trunc i128 %shr to i32
  ret i32 %ext

; CHECK-LABEL: @trunc_bitcast2(
; CHECK-NEXT:  %bc1 = bitcast <2 x i64> %v to <4 x i32>
; CHECK-NEXT:  %ext = extractelement <4 x i32> %bc1, i32 2
; CHECK-NEXT:  ret i32 %ext
}

; The right shift is optional.

define i32 @trunc_bitcast3(<4 x i32> %v) {
  %bc = bitcast <4 x i32> %v to i128
  %ext = trunc i128 %bc to i32
  ret i32 %ext

; CHECK-LABEL: @trunc_bitcast3(
; CHECK-NEXT:  %ext = extractelement <4 x i32> %v, i32 0
; CHECK-NEXT:  ret i32 %ext
}

; CHECK-LABEL: @trunc_shl_31_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 31
define i32 @trunc_shl_31_i32_i64(i64 %val) {
  %shl = shl i64 %val, 31
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_nsw_31_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 31
define i32 @trunc_shl_nsw_31_i32_i64(i64 %val) {
  %shl = shl nsw i64 %val, 31
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_nuw_31_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 31
define i32 @trunc_shl_nuw_31_i32_i64(i64 %val) {
  %shl = shl nuw i64 %val, 31
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_nsw_nuw_31_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 31
define i32 @trunc_shl_nsw_nuw_31_i32_i64(i64 %val) {
  %shl = shl nsw nuw i64 %val, 31
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_15_i16_i64(
; CHECK: %val.tr = trunc i64 %val to i16
; CHECK-NEXT: shl i16 %val.tr, 15
define i16 @trunc_shl_15_i16_i64(i64 %val) {
  %shl = shl i64 %val, 15
  %trunc = trunc i64 %shl to i16
  ret i16 %trunc
}

; CHECK-LABEL: @trunc_shl_15_i16_i32(
; CHECK: %val.tr = trunc i32 %val to i16
; CHECK-NEXT: shl i16 %val.tr, 15
define i16 @trunc_shl_15_i16_i32(i32 %val) {
  %shl = shl i32 %val, 15
  %trunc = trunc i32 %shl to i16
  ret i16 %trunc
}

; CHECK-LABEL: @trunc_shl_7_i8_i64(
; CHECK: %val.tr = trunc i64 %val to i8
; CHECK-NEXT: shl i8 %val.tr, 7
define i8 @trunc_shl_7_i8_i64(i64 %val) {
  %shl = shl i64 %val, 7
  %trunc = trunc i64 %shl to i8
  ret i8 %trunc
}

; CHECK-LABEL: @trunc_shl_1_i2_i64(
; CHECK: shl i64 %val, 1
; CHECK-NEXT: trunc i64 %shl to i2
define i2 @trunc_shl_1_i2_i64(i64 %val) {
  %shl = shl i64 %val, 1
  %trunc = trunc i64 %shl to i2
  ret i2 %trunc
}

; CHECK-LABEL: @trunc_shl_1_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 1
define i32 @trunc_shl_1_i32_i64(i64 %val) {
  %shl = shl i64 %val, 1
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_16_i32_i64(
; CHECK: %val.tr = trunc i64 %val to i32
; CHECK-NEXT: shl i32 %val.tr, 16
define i32 @trunc_shl_16_i32_i64(i64 %val) {
  %shl = shl i64 %val, 16
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_33_i32_i64(
; CHECK: ret i32 0
define i32 @trunc_shl_33_i32_i64(i64 %val) {
  %shl = shl i64 %val, 33
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; CHECK-LABEL: @trunc_shl_32_i32_i64(
; CHECK: ret i32 0
define i32 @trunc_shl_32_i32_i64(i64 %val) {
  %shl = shl i64 %val, 32
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; TODO: Should be able to handle vectors
; CHECK-LABEL: @trunc_shl_16_v2i32_v2i64(
; CHECK: shl <2 x i64>
define <2 x i32> @trunc_shl_16_v2i32_v2i64(<2 x i64> %val) {
  %shl = shl <2 x i64> %val, <i64 16, i64 16>
  %trunc = trunc <2 x i64> %shl to <2 x i32>
  ret <2 x i32> %trunc
}

; CHECK-LABEL: @trunc_shl_nosplat_v2i32_v2i64(
; CHECK: shl <2 x i64>
define <2 x i32> @trunc_shl_nosplat_v2i32_v2i64(<2 x i64> %val) {
  %shl = shl <2 x i64> %val, <i64 15, i64 16>
  %trunc = trunc <2 x i64> %shl to <2 x i32>
  ret <2 x i32> %trunc
}

; CHECK-LABEL: @trunc_shl_31_i32_i64_multi_use(
; CHECK: shl i64 %val, 31
; CHECK-NOT: shl i32
; CHECK: trunc i64 %shl to i32
; CHECK-NOT: shl i32
define void @trunc_shl_31_i32_i64_multi_use(i64 %val, i32 addrspace(1)* %ptr0, i64 addrspace(1)* %ptr1) {
  %shl = shl i64 %val, 31
  %trunc = trunc i64 %shl to i32
  store volatile i32 %trunc, i32 addrspace(1)* %ptr0
  store volatile i64 %shl, i64 addrspace(1)* %ptr1
  ret void
}

; CHECK-LABEL: @trunc_shl_lshr_infloop(
; CHECK-NEXT: %tmp0 = lshr i64 %arg, 1
; CHECK-NEXT: %tmp1 = shl i64 %tmp0, 2
; CHECK-NEXT: %tmp2 = trunc i64 %tmp1 to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_lshr_infloop(i64 %arg) {
  %tmp0 = lshr i64 %arg, 1
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_ashr_infloop(
; CHECK-NEXT: %tmp0 = ashr i64 %arg, 3
; CHECK-NEXT: %tmp1 = shl nsw i64 %tmp0, 2
; CHECK-NEXT: %tmp2 = trunc i64 %tmp1 to i32
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_ashr_infloop(i64 %arg) {
  %tmp0 = ashr i64 %arg, 3
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_shl_infloop(
; CHECK-NEXT: %arg.tr = trunc i64 %arg to i32
; CHECK-NEXT: %tmp2 = shl i32 %arg.tr, 3
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_shl_infloop(i64 %arg) {
  %tmp0 = shl i64 %arg, 1
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_lshr_var(
; CHECK-NEXT: %tmp0 = lshr i64 %arg, %val
; CHECK-NEXT: %tmp0.tr = trunc i64 %tmp0 to i32
; CHECK-NEXT: %tmp2 = shl i32 %tmp0.tr, 2
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_lshr_var(i64 %arg, i64 %val) {
  %tmp0 = lshr i64 %arg, %val
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_ashr_var(
; CHECK-NEXT: %tmp0 = ashr i64 %arg, %val
; CHECK-NEXT: %tmp0.tr = trunc i64 %tmp0 to i32
; CHECK-NEXT: %tmp2 = shl i32 %tmp0.tr, 2
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_ashr_var(i64 %arg, i64 %val) {
  %tmp0 = ashr i64 %arg, %val
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_shl_var(
; CHECK-NEXT: %tmp0 = shl i64 %arg, %val
; CHECK-NEXT: %tmp0.tr = trunc i64 %tmp0 to i32
; CHECK-NEXT: %tmp2 = shl i32 %tmp0.tr, 2
; CHECK-NEXT: ret i32 %tmp2
define i32 @trunc_shl_shl_var(i64 %arg, i64 %val) {
  %tmp0 = shl i64 %arg, %val
  %tmp1 = shl i64 %tmp0, 2
  %tmp2 = trunc i64 %tmp1 to i32
  ret i32 %tmp2
}

; CHECK-LABEL: @trunc_shl_v8i15_v8i32_15(
; CHECK: %shl = shl <8 x i32> %a, <i32 15,
; CHECK: trunc <8 x i32> %shl to <8 x i16>
define <8 x i16> @trunc_shl_v8i15_v8i32_15(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %conv = trunc <8 x i32> %shl to <8 x i16>
  ret <8 x i16> %conv
}

; CHECK-LABEL: @trunc_shl_v8i16_v8i32_16(
; CHECK: %shl = shl <8 x i32> %a, <i32 16
; CHECK: trunc <8 x i32> %shl to <8 x i16>
define <8 x i16> @trunc_shl_v8i16_v8i32_16(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %conv = trunc <8 x i32> %shl to <8 x i16>
  ret <8 x i16> %conv
}

; CHECK-LABEL: @trunc_shl_v8i16_v8i32_17(
; CHECK: %shl = shl <8 x i32> %a, <i32 17
; CHECK: %conv = trunc <8 x i32> %shl to <8 x i16>
define <8 x i16> @trunc_shl_v8i16_v8i32_17(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17>
  %conv = trunc <8 x i32> %shl to <8 x i16>
  ret <8 x i16> %conv
}

; CHECK-LABEL: @trunc_shl_v8i16_v8i32_4(
; CHECK: %shl = shl <8 x i32> %a, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
; CHECK: trunc <8 x i32> %shl to <8 x i16>
define <8 x i16> @trunc_shl_v8i16_v8i32_4(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %conv = trunc <8 x i32> %shl to <8 x i16>
  ret <8 x i16> %conv
}
