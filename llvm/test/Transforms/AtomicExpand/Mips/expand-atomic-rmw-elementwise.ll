; RUN: opt -S -mtriple=mips64-mti-linux-gnu -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s

; MIPS uses TargetLowering's default AtomicRMW expansion, so an elementwise
; atomicrmw is scalarized.
define <2 x i32> @atomicrmw_add_elementwise(ptr %ptr, <2 x i32> %value) {
; CHECK-LABEL: define <2 x i32> @atomicrmw_add_elementwise(
; CHECK-SAME: ptr [[PTR:%.*]], <2 x i32> [[VALUE:%.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LO_VALUE:%.*]] = extractelement <2 x i32> [[VALUE]], i64 0
; CHECK-NEXT:   [[HI_VALUE:%.*]] = extractelement <2 x i32> [[VALUE]], i64 1
; CHECK-NEXT:   [[HI_PTR:%.*]] = getelementptr inbounds i32, ptr [[PTR]], i64 1
; CHECK-NEXT:   [[LO_OLD:%.*]] = atomicrmw add ptr [[PTR]], i32 [[LO_VALUE]] monotonic, align 8
; CHECK-NEXT:   [[HI_OLD:%.*]] = atomicrmw add ptr [[HI_PTR]], i32 [[HI_VALUE]] monotonic, align 4
; CHECK-NEXT:   [[RESULT:%.*]] = insertelement <2 x i32> poison, i32 [[LO_OLD]], i64 0
; CHECK-NEXT:   [[RESULT1:%.*]] = insertelement <2 x i32> [[RESULT]], i32 [[HI_OLD]], i64 1
; CHECK-NEXT:   ret <2 x i32> [[RESULT1]]
;
entry:
  %old = atomicrmw elementwise add ptr %ptr, <2 x i32> %value monotonic
  ret <2 x i32> %old
}
