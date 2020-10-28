; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve.fp -tail-predication=force-enabled-no-reductions -o - %s | FileCheck %s

define arm_aapcs_vfpcc <16 x i8> @vcmp_vpst_combination(<16 x i8>* %pSrc, i16 zeroext %blockSize, i8* nocapture %pResult, i32* nocapture %pIndex) {
; CHECK-LABEL: vcmp_vpst_combination:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    vmov.i8 q0, #0x7f
; CHECK-NEXT:    dlstp.8 lr, r1
; CHECK-NEXT:  .LBB0_1: @ %do.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrb.u8 q1, [r0]
; CHECK-NEXT:    vpt.s8 ge, q0, q1
; CHECK-NEXT:    vmovt q0, q1
; CHECK-NEXT:    letp lr, .LBB0_1
; CHECK-NEXT:  @ %bb.2: @ %do.end
; CHECK-NEXT:    pop {r7, pc}
entry:
  %conv = zext i16 %blockSize to i32
  %0 = tail call { <16 x i8>, i32 } @llvm.arm.mve.vidup.v16i8(i32 0, i32 1)
  %1 = extractvalue { <16 x i8>, i32 } %0, 0
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %indexVec.0 = phi <16 x i8> [ %1, %entry ], [ %add, %do.body ]
  %curExtremIdxVec.0 = phi <16 x i8> [ zeroinitializer, %entry ], [ %6, %do.body ]
  %curExtremValVec.0 = phi <16 x i8> [ <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127>, %entry ], [ %6, %do.body ]
  %blkCnt.0 = phi i32 [ %conv, %entry ], [ %sub2, %do.body ]
  %2 = tail call <16 x i1> @llvm.arm.mve.vctp8(i32 %blkCnt.0)
  %3 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %pSrc, i32 1, <16 x i1> %2, <16 x i8> zeroinitializer)
  %4 = icmp sle <16 x i8> %3, %curExtremValVec.0
  %5 = and <16 x i1> %4, %2
  %6 = tail call <16 x i8> @llvm.arm.mve.orr.predicated.v16i8.v16i1(<16 x i8> %3, <16 x i8> %3, <16 x i1> %5, <16 x i8> %curExtremValVec.0)
  %add = add <16 x i8> %indexVec.0, <i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16>
  %sub2 = add nsw i32 %blkCnt.0, -16
  %cmp = icmp sgt i32 %blkCnt.0, 16
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  ret <16 x i8> %6
}

declare { <16 x i8>, i32 } @llvm.arm.mve.vidup.v16i8(i32, i32)

declare <16 x i1> @llvm.arm.mve.vctp8(i32)

declare <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>*, i32 immarg, <16 x i1>, <16 x i8>)

declare <16 x i8> @llvm.arm.mve.orr.predicated.v16i8.v16i1(<16 x i8>, <16 x i8>, <16 x i1>, <16 x i8>)
