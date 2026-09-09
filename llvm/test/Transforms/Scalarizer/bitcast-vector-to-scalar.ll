; RUN: opt %s -passes='scalarizer,dce' -S --data-layout=e | FileCheck %s --check-prefix=LE
; RUN: opt %s -passes='scalarizer,dce' -S --data-layout=E | FileCheck %s --check-prefix=BE

declare <2 x i16> @llvm.ctpop.v2i16(<2 x i16>)

define i32 @bitcast_vector_to_scalar(<2 x i16> %x) {
; LE-LABEL: define i32 @bitcast_vector_to_scalar(
; LE: [[LO:%.*]] = call i16 @llvm.ctpop.i16
; LE: [[HI:%.*]] = call i16 @llvm.ctpop.i16
; LE: [[LO_EXT:%.*]] = zext i16 [[LO]] to i32
; LE: [[HI_EXT:%.*]] = zext i16 [[HI]] to i32
; LE: [[HI_SHIFT:%.*]] = shl i32 [[HI_EXT]], 16
; LE: [[RESULT:%.*]] = or i32 [[LO_EXT]], [[HI_SHIFT]]
; LE-NOT: insertelement
; LE-NOT: bitcast
; LE: ret i32 [[RESULT]]
;
; BE-LABEL: define i32 @bitcast_vector_to_scalar(
; BE: [[HI:%.*]] = call i16 @llvm.ctpop.i16
; BE: [[LO:%.*]] = call i16 @llvm.ctpop.i16
; BE: [[HI_EXT:%.*]] = zext i16 [[HI]] to i32
; BE: [[HI_SHIFT:%.*]] = shl i32 [[HI_EXT]], 16
; BE: [[LO_EXT:%.*]] = zext i16 [[LO]] to i32
; BE: [[RESULT:%.*]] = or i32 [[HI_SHIFT]], [[LO_EXT]]
; BE-NOT: insertelement
; BE-NOT: bitcast
; BE: ret i32 [[RESULT]]
entry:
  %count = call <2 x i16> @llvm.ctpop.v2i16(<2 x i16> %x)
  %result = bitcast <2 x i16> %count to i32
  ret i32 %result
}
