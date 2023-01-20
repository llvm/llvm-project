
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mcpu=neoverse-v1 -O3 -opaque-pointers -aarch64-sve-vector-bits-min=256 -verify-machineinstrs | FileCheck %s --check-prefixes=SVE256
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mcpu=neoverse-v1 -O3 -opaque-pointers -aarch64-sve-vector-bits-min=128 -verify-machineinstrs | FileCheck %s --check-prefixes=NEON
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mcpu=neoverse-n1 -O3 -opaque-pointers -verify-machineinstrs | FileCheck %s --check-prefixes=NEON
; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -mcpu=neoverse-v2 -O3 -opaque-pointers -verify-machineinstrs | FileCheck %s --check-prefixes=NEON

define internal i32 @test(ptr nocapture readonly %p1, i32 %i1, ptr nocapture readonly %p2, i32 %i2) {
; SVE256-LABEL: test:
; SVE256:       ld1b	{ z0.h }, p0/z,
; SVE256:       ld1b	{ z1.h }, p0/z,
; SVE256:       sub	z0.h, z0.h, z1.h
; SVE256-NEXT:  sunpklo	z1.s, z0.h
; SVE256-NEXT:  ext	z0.b, z0.b, z0.b, #16
; SVE256-NEXT:  sunpklo	z0.s, z0.h
; SVE256-NEXT:  add	z0.s, z1.s, z0.s
; SVE256-NEXT:  uaddv	d0, p1, z0.s

; NEON-LABEL: test:
; NEON:         tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON-NEXT:    tbl
; NEON:         addv


L.entry:
  br label %L1

L1:                                          ; preds = %L1, %L.entry
  %a = phi i32 [ 16, %L.entry ], [ %14, %L1 ]
  %b = phi i32 [ 0, %L.entry ], [ %13, %L1 ]
  %i = phi i32 [ 0, %L.entry ], [ %12, %L1 ]
  %0 = mul i32 %b, %i1
  %1 = sext i32 %0 to i64
  %2 = getelementptr i8, ptr %p1, i64 %1
  %3 = mul i32 %b, %i2
  %4 = sext i32 %3 to i64
  %5 = getelementptr i8, ptr %p2, i64 %4
  %6 = load <16 x i8>, ptr %2, align 1
  %7 = zext <16 x i8> %6 to <16 x i32>
  %8 = load <16 x i8>, ptr %5, align 1
  %9 = zext <16 x i8> %8 to <16 x i32>
  %10 = sub nsw <16 x i32> %7, %9
  %11 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %10)
  %12 = add i32 %11, %i
  %13 = add nuw nsw i32 %b, 1
  %14 = add nsw i32 %a, -1
  %.not = icmp eq i32 %14, 0
  br i1 %.not, label %L2, label %L1

L2:                                          ; preds = %L1
  ret i32 %12
}

declare  i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)
