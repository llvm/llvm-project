; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

define i32 @vmin_u8x8(<8 x i8> %a) nounwind ssp {
; CHECK-LABEL: vmin_u8x8:
; CHECK: uminv.8b        b[[REG:[0-9]+]], v0
; CHECK: fmov    [[REG2:w[0-9]+]], s[[REG]]
; CHECK-NOT: and
; CHECK: cbz     [[REG2]],
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8> %a) nounwind
  %tmp = trunc i32 %vminv.i to i8
  %tobool = icmp eq i8 %tmp, 0
  br i1 %tobool, label %return, label %if.then

if.then:
  %call1 = tail call i32 @bar() nounwind
  br label %return

return:
  %retval.0 = phi i32 [ %call1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

declare i32 @bar(...)

define i32 @vmin_u4x16(<4 x i16> %a) nounwind ssp {
; CHECK-LABEL: vmin_u4x16:
; CHECK: uminv.4h        h[[REG:[0-9]+]], v0
; CHECK: fmov    [[REG2:w[0-9]+]], s[[REG]]
; CHECK-NOT: and
; CHECK: cbz     [[REG2]],
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16> %a) nounwind
  %tmp = trunc i32 %vminv.i to i16
  %tobool = icmp eq i16 %tmp, 0
  br i1 %tobool, label %return, label %if.then

if.then:
  %call1 = tail call i32 @bar() nounwind
  br label %return

return:
  %retval.0 = phi i32 [ %call1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

define i32 @vmin_u8x16(<8 x i16> %a) nounwind ssp {
; CHECK-LABEL: vmin_u8x16:
; CHECK: uminv.8h        h[[REG:[0-9]+]], v0
; CHECK: fmov    [[REG2:w[0-9]+]], s[[REG]]
; CHECK-NOT: and
; CHECK: cbz     [[REG2]],
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16> %a) nounwind
  %tmp = trunc i32 %vminv.i to i16
  %tobool = icmp eq i16 %tmp, 0
  br i1 %tobool, label %return, label %if.then

if.then:
  %call1 = tail call i32 @bar() nounwind
  br label %return

return:
  %retval.0 = phi i32 [ %call1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

define i32 @vmin_u16x8(<16 x i8> %a) nounwind ssp {
; CHECK-LABEL: vmin_u16x8:
; CHECK: uminv.16b        b[[REG:[0-9]+]], v0
; CHECK: fmov     [[REG2:w[0-9]+]], s[[REG]]
; CHECK-NOT: and
; CHECK: cbz     [[REG2]],
entry:
  %vminv.i = tail call i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8> %a) nounwind
  %tmp = trunc i32 %vminv.i to i8
  %tobool = icmp eq i8 %tmp, 0
  br i1 %tobool, label %return, label %if.then

if.then:
  %call1 = tail call i32 @bar() nounwind
  br label %return

return:
  %retval.0 = phi i32 [ %call1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

define <8 x i8> @test_vminv_u8_used_by_laneop(<8 x i8> %a1, <8 x i8> %a2) {
; CHECK-LABEL: test_vminv_u8_used_by_laneop:
; CHECK: uminv.8b b[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: mov.b v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8> %a2)
  %1 = trunc i32 %0 to i8
  %2 = insertelement <8 x i8> %a1, i8 %1, i32 3
  ret <8 x i8> %2
}

define <4 x i16> @test_vminv_u16_used_by_laneop(<4 x i16> %a1, <4 x i16> %a2) {
; CHECK-LABEL: test_vminv_u16_used_by_laneop:
; CHECK: uminv.4h h[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: mov.h v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16> %a2)
  %1 = trunc i32 %0 to i16
  %2 = insertelement <4 x i16> %a1, i16 %1, i32 3
  ret <4 x i16> %2
}

define <2 x i32> @test_vminv_u32_used_by_laneop(<2 x i32> %a1, <2 x i32> %a2) {
; CHECK-LABEL: test_vminv_u32_used_by_laneop:
; CHECK: uminp.2s v[[REGNUM:[0-9]+]], v1, v1
; CHECK-NEXT: mov.s v0[1], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v2i32(<2 x i32> %a2)
  %1 = insertelement <2 x i32> %a1, i32 %0, i32 1
  ret <2 x i32> %1
}

define <16 x i8> @test_vminvq_u8_used_by_laneop(<16 x i8> %a1, <16 x i8> %a2) {
; CHECK-LABEL: test_vminvq_u8_used_by_laneop:
; CHECK: uminv.16b b[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: mov.b v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8> %a2)
  %1 = trunc i32 %0 to i8
  %2 = insertelement <16 x i8> %a1, i8 %1, i32 3
  ret <16 x i8> %2
}

define <8 x i16> @test_vminvq_u16_used_by_laneop(<8 x i16> %a1, <8 x i16> %a2) {
; CHECK-LABEL: test_vminvq_u16_used_by_laneop:
; CHECK: uminv.8h h[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: mov.h v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16> %a2)
  %1 = trunc i32 %0 to i16
  %2 = insertelement <8 x i16> %a1, i16 %1, i32 3
  ret <8 x i16> %2
}

define <4 x i32> @test_vminvq_u32_used_by_laneop(<4 x i32> %a1, <4 x i32> %a2) {
; CHECK-LABEL: test_vminvq_u32_used_by_laneop:
; CHECK: uminv.4s s[[REGNUM:[0-9]+]], v1
; CHECK-NEXT: mov.s v0[3], v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %0 = tail call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %a2)
  %1 = insertelement <4 x i32> %a1, i32 %0, i32 3
  ret <4 x i32> %1
}
declare i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8>) nounwind readnone
declare i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16>) nounwind readnone
declare i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16>) nounwind readnone
declare i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8>) nounwind readnone
declare i32 @llvm.aarch64.neon.uminv.i32.v2i32(<2 x i32>) nounwind readnone
declare i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32>) nounwind readnone
