; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=dynamic-no-pic -frame-pointer=all | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all | FileCheck %s --check-prefix=PIC
; rdar://7353541
; rdar://7354376

@GV = external global i32                         ; <ptr> [#uses=2]

define void @t1(ptr nocapture %vals, i32 %c) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: bxeq lr

  %0 = icmp eq i32 %c, 0                          ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb.nph

bb.nph:                                           ; preds = %entry
; CHECK: movw r[[R2:[0-9]+]], :lower16:L_GV$non_lazy_ptr
; CHECK: movt r[[R2]], :upper16:L_GV$non_lazy_ptr
; CHECK: ldr{{(.w)?}} r[[R2b:[0-9]+]], [r[[R2]]
; CHECK: ldr{{.*}}, [r[[R2b]]
; CHECK: LBB0_
; CHECK-NOT: LCPI0_0:

; PIC: movw r[[R2:[0-9]+]], :lower16:(L_GV$non_lazy_ptr-(LPC0_0+4))
; PIC: movt r[[R2]], :upper16:(L_GV$non_lazy_ptr-(LPC0_0+4))
; PIC: add r[[R2]], pc
; PIC: ldr{{(.w)?}} r[[R2b:[0-9]+]], [r[[R2]]
; PIC: ldr{{.*}}, [r[[R2b]]
; PIC: LBB0_
; PIC-NOT: LCPI0_0:
; PIC: .section
  %.pre = load i32, ptr @GV, align 4                  ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %1 = phi i32 [ %.pre, %bb.nph ], [ %3, %bb ]    ; <i32> [#uses=1]
  %i.03 = phi i32 [ 0, %bb.nph ], [ %4, %bb ]     ; <i32> [#uses=2]
  %scevgep = getelementptr i32, ptr %vals, i32 %i.03  ; <ptr> [#uses=1]
  %2 = load i32, ptr %scevgep, align 4                ; <i32> [#uses=1]
  %3 = add nsw i32 %1, %2                         ; <i32> [#uses=2]
  store i32 %3, ptr @GV, align 4
  %4 = add i32 %i.03, 1                           ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %4, %c                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

; rdar://8001136
define void @t2(ptr %ptr1, ptr %ptr2) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: vmov.f32 q{{.*}}, #1.000000e+00
  br i1 undef, label %bb1, label %bb2

bb1:
; CHECK: %bb1
  %indvar = phi i32 [ %indvar.next, %bb1 ], [ 0, %entry ]
  %tmp1 = shl i32 %indvar, 2
  %gep1 = getelementptr i8, ptr %ptr1, i32 %tmp1
  %tmp2 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr %gep1, i32 1)
  %tmp3 = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float> %tmp2)
  %gep2 = getelementptr i8, ptr %ptr2, i32 %tmp1
  call void @llvm.arm.neon.vst1.p0.v4f32(ptr %gep2, <4 x float> %tmp3, i32 1)
  %indvar.next = add i32 %indvar, 1
  %cond = icmp eq i32 %indvar.next, 10
  br i1 %cond, label %bb2, label %bb1

bb2:
  ret void
}

; CHECK-NOT: LCPI1_0:

declare <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.p0.v4f32(ptr, <4 x float>, i32) nounwind

declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>) nounwind readnone

; rdar://8241368
; isel should not fold immediate into eor's which would have prevented LICM.
define zeroext i16 @t3(i8 zeroext %data, i16 zeroext %crc) nounwind readnone {
; CHECK-LABEL: t3:
bb.nph:
; CHECK: bb.nph
; CHECK: movw {{(r[0-9]+)|(lr)}}, #32768
; CHECK: movs {{(r[0-9]+)|(lr)}}, #0
; CHECK: movw [[REGISTER:(r[0-9]+)|(lr)]], #16386
; CHECK: movt {{(r[0-9]+)|(lr)}}, #65535
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
; CHECK: bb
; CHECK: eor.w
; CHECK: eorne.w {{(r[0-9])|(lr)}}, {{(r[0-9])|(lr)}}, [[REGISTER]]
; CHECK-NOT: eor
  %data_addr.013 = phi i8 [ %data, %bb.nph ], [ %8, %bb ] ; <i8> [#uses=2]
  %crc_addr.112 = phi i16 [ %crc, %bb.nph ], [ %crc_addr.2, %bb ] ; <i16> [#uses=3]
  %i.011 = phi i8 [ 0, %bb.nph ], [ %7, %bb ]     ; <i8> [#uses=1]
  %0 = trunc i16 %crc_addr.112 to i8              ; <i8> [#uses=1]
  %1 = xor i8 %data_addr.013, %0                  ; <i8> [#uses=1]
  %2 = and i8 %1, 1                               ; <i8> [#uses=1]
  %3 = icmp eq i8 %2, 0                           ; <i1> [#uses=2]
  %4 = xor i16 %crc_addr.112, 16386               ; <i16> [#uses=1]
  %crc_addr.0 = select i1 %3, i16 %crc_addr.112, i16 %4 ; <i16> [#uses=1]
  %5 = lshr i16 %crc_addr.0, 1                    ; <i16> [#uses=2]
  %6 = or i16 %5, -32768                          ; <i16> [#uses=1]
  %crc_addr.2 = select i1 %3, i16 %5, i16 %6      ; <i16> [#uses=2]
  %7 = add i8 %i.011, 1                           ; <i8> [#uses=2]
  %8 = lshr i8 %data_addr.013, 1                  ; <i8> [#uses=1]
  %exitcond = icmp eq i8 %7, 8                    ; <i1> [#uses=1]
  br i1 %exitcond, label %bb8, label %bb

bb8:                                              ; preds = %bb
  ret i16 %crc_addr.2
}
