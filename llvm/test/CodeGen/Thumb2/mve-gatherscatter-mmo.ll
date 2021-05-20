; RUN: llc -mtriple=thumbv8.1m.main -mattr=+mve.fp -stop-after=finalize-isel -o - %s | FileCheck %s

define arm_aapcs_vfpcc <8 x i16> @test_vldrbq_gather_offset_s16(i8* %base, <8 x i16> %offset) {
; CHECK-LABEL: name: test_vldrbq_gather_offset_s16
; CHECK: early-clobber %2:mqpr = MVE_VLDRBS16_rq %0, %1, 0, $noreg :: (load (s64), align 1)
entry:
  %0 = call <8 x i16> @llvm.arm.mve.vldr.gather.offset.v8i16.p0i8.v8i16(i8* %base, <8 x i16> %offset, i32 8, i32 0, i32 0)
  ret <8 x i16> %0
}

define arm_aapcs_vfpcc <4 x i32> @test_vldrbq_gather_offset_z_s32(i8* %base, <4 x i32> %offset, i16 zeroext %p) {
; CHECK-LABEL: name: test_vldrbq_gather_offset_z_s32
; CHECK: early-clobber %4:mqpr = MVE_VLDRBS32_rq %0, %1, 1, killed %3 :: (load (s32), align 1)
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = call <4 x i32> @llvm.arm.mve.vldr.gather.offset.predicated.v4i32.p0i8.v4i32.v4i1(i8* %base, <4 x i32> %offset, i32 8, i32 0, i32 0, <4 x i1> %1)
  ret <4 x i32> %2
}

define arm_aapcs_vfpcc <2 x i64> @test_vldrdq_gather_base_s64(<2 x i64> %addr) {
; CHECK-LABEL: name: test_vldrdq_gather_base_s64
; CHECK: early-clobber %1:mqpr = MVE_VLDRDU64_qi %0, 616, 0, $noreg :: (load (s128), align 1)
entry:
  %0 = call <2 x i64> @llvm.arm.mve.vldr.gather.base.v2i64.v2i64(<2 x i64> %addr, i32 616)
  ret <2 x i64> %0
}

define arm_aapcs_vfpcc <4 x float> @test_vldrwq_gather_base_z_f32(<4 x i32> %addr, i16 zeroext %p) {
; CHECK-LABEL: name: test_vldrwq_gather_base_z_f32
; CHECK: early-clobber %3:mqpr = MVE_VLDRWU32_qi %0, -300, 1, killed %2 :: (load (s128), align 1)
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  %2 = call <4 x float> @llvm.arm.mve.vldr.gather.base.predicated.v4f32.v4i32.v4i1(<4 x i32> %addr, i32 -300, <4 x i1> %1)
  ret <4 x float> %2
}

define arm_aapcs_vfpcc <2 x i64> @test_vldrdq_gather_base_wb_s64(<2 x i64>* %addr) {
; CHECK-LABEL: name: test_vldrdq_gather_base_wb_s64
; CHECK: %2:mqpr, early-clobber %3:mqpr = MVE_VLDRDU64_qi_pre %1, 576, 0, $noreg :: (load (s128), align 1)
entry:
  %0 = load <2 x i64>, <2 x i64>* %addr, align 8
  %1 = call { <2 x i64>, <2 x i64> } @llvm.arm.mve.vldr.gather.base.wb.v2i64.v2i64(<2 x i64> %0, i32 576)
  %2 = extractvalue { <2 x i64>, <2 x i64> } %1, 1
  store <2 x i64> %2, <2 x i64>* %addr, align 8
  %3 = extractvalue { <2 x i64>, <2 x i64> } %1, 0
  ret <2 x i64> %3
}

define arm_aapcs_vfpcc <4 x float> @test_vldrwq_gather_base_wb_z_f32(<4 x i32>* %addr, i16 zeroext %p) {
; CHECK-LABEL: name: test_vldrwq_gather_base_wb_z_f32
; CHECK: %4:mqpr, early-clobber %5:mqpr = MVE_VLDRWU32_qi_pre %3, -352, 1, killed %2 :: (load (s128), align 1)
entry:
  %0 = load <4 x i32>, <4 x i32>* %addr, align 8
  %1 = zext i16 %p to i32
  %2 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %1)
  %3 = call { <4 x float>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.predicated.v4f32.v4i32.v4i1(<4 x i32> %0, i32 -352, <4 x i1> %2)
  %4 = extractvalue { <4 x float>, <4 x i32> } %3, 1
  store <4 x i32> %4, <4 x i32>* %addr, align 8
  %5 = extractvalue { <4 x float>, <4 x i32> } %3, 0
  ret <4 x float> %5
}


define arm_aapcs_vfpcc void @test_vstrbq_scatter_offset_s32(i8* %base, <4 x i32> %offset, <4 x i32> %value) {
; CHECK-LABEL: name: test_vstrbq_scatter_offset_s32
; CHECK: MVE_VSTRB32_rq %2, %0, %1, 0, $noreg :: (store (s32), align 1)
entry:
  call void @llvm.arm.mve.vstr.scatter.offset.p0i8.v4i32.v4i32(i8* %base, <4 x i32> %offset, <4 x i32> %value, i32 8, i32 0)
  ret void
}

define arm_aapcs_vfpcc void @test_vstrbq_scatter_offset_p_s8(i8* %base, <16 x i8> %offset, <16 x i8> %value, i16 zeroext %p) {
; CHECK-LABEL: name: test_vstrbq_scatter_offset_p_s8
; CHECK: MVE_VSTRB8_rq %2, %0, %1, 1, killed %4 :: (store (s128), align 1)
entry:
  %0 = zext i16 %p to i32
  %1 = call <16 x i1> @llvm.arm.mve.pred.i2v.v16i1(i32 %0)
  call void @llvm.arm.mve.vstr.scatter.offset.predicated.p0i8.v16i8.v16i8.v16i1(i8* %base, <16 x i8> %offset, <16 x i8> %value, i32 8, i32 0, <16 x i1> %1)
  ret void
}

define arm_aapcs_vfpcc void @test_vstrdq_scatter_base_u64(<2 x i64> %addr, <2 x i64> %value) {
; CHECK-LABEL: name: test_vstrdq_scatter_base_u64
; CHECK: MVE_VSTRD64_qi %1, %0, -472, 0, $noreg :: (store (s128), align 1)
entry:
  call void @llvm.arm.mve.vstr.scatter.base.v2i64.v2i64(<2 x i64> %addr, i32 -472, <2 x i64> %value)
  ret void
}

define arm_aapcs_vfpcc void @test_vstrdq_scatter_base_p_s64(<2 x i64> %addr, <2 x i64> %value, i16 zeroext %p) {
; CHECK-LABEL: name: test_vstrdq_scatter_base_p_s64
; CHECK: MVE_VSTRD64_qi %1, %0, 888, 1, killed %3 :: (store (s128), align 1)
entry:
  %0 = zext i16 %p to i32
  %1 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %0)
  call void @llvm.arm.mve.vstr.scatter.base.predicated.v2i64.v2i64.v4i1(<2 x i64> %addr, i32 888, <2 x i64> %value, <4 x i1> %1)
  ret void
}

define arm_aapcs_vfpcc void @test_vstrdq_scatter_base_wb_s64(<2 x i64>* %addr, <2 x i64> %value) {
; CHECK-LABEL: name: test_vstrdq_scatter_base_wb_s64
; CHECK: %3:mqpr = MVE_VSTRD64_qi_pre %1, %2, 208, 0, $noreg :: (store (s128), align 1)
entry:
  %0 = load <2 x i64>, <2 x i64>* %addr, align 8
  %1 = call <2 x i64> @llvm.arm.mve.vstr.scatter.base.wb.v2i64.v2i64(<2 x i64> %0, i32 208, <2 x i64> %value)
  store <2 x i64> %1, <2 x i64>* %addr, align 8
  ret void
}

define arm_aapcs_vfpcc void @test_vstrdq_scatter_base_wb_p_s64(<2 x i64>* %addr, <2 x i64> %value, i16 zeroext %p) {
; CHECK-LABEL: name: test_vstrdq_scatter_base_wb_p_s64
; CHECK: %5:mqpr = MVE_VSTRD64_qi_pre %1, %3, 248, 1, killed %4 :: (store (s128), align 1)
entry:
  %0 = load <2 x i64>, <2 x i64>* %addr, align 8
  %1 = zext i16 %p to i32
  %2 = call <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32 %1)
  %3 = call <2 x i64> @llvm.arm.mve.vstr.scatter.base.wb.predicated.v2i64.v2i64.v4i1(<2 x i64> %0, i32 248, <2 x i64> %value, <4 x i1> %2)
  store <2 x i64> %3, <2 x i64>* %addr, align 8
  ret void
}

declare <16 x i1> @llvm.arm.mve.pred.i2v.v16i1(i32)
declare <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32)
declare <8 x i16> @llvm.arm.mve.vldr.gather.offset.v8i16.p0i8.v8i16(i8*, <8 x i16>, i32, i32, i32)
declare <4 x i32> @llvm.arm.mve.vldr.gather.offset.predicated.v4i32.p0i8.v4i32.v4i1(i8*, <4 x i32>, i32, i32, i32, <4 x i1>)
declare <2 x i64> @llvm.arm.mve.vldr.gather.base.v2i64.v2i64(<2 x i64>, i32)
declare <4 x float> @llvm.arm.mve.vldr.gather.base.predicated.v4f32.v4i32.v4i1(<4 x i32>, i32, <4 x i1>)
declare { <2 x i64>, <2 x i64> } @llvm.arm.mve.vldr.gather.base.wb.v2i64.v2i64(<2 x i64>, i32)
declare { <4 x float>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.predicated.v4f32.v4i32.v4i1(<4 x i32>, i32, <4 x i1>)
declare void @llvm.arm.mve.vstr.scatter.offset.p0i8.v4i32.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32)
declare void @llvm.arm.mve.vstr.scatter.offset.predicated.p0i8.v16i8.v16i8.v16i1(i8*, <16 x i8>, <16 x i8>, i32, i32, <16 x i1>)
declare void @llvm.arm.mve.vstr.scatter.base.v2i64.v2i64(<2 x i64>, i32, <2 x i64>)
declare void @llvm.arm.mve.vstr.scatter.base.predicated.v2i64.v2i64.v4i1(<2 x i64>, i32, <2 x i64>, <4 x i1>)
declare <2 x i64> @llvm.arm.mve.vstr.scatter.base.wb.v2i64.v2i64(<2 x i64>, i32, <2 x i64>)
declare <2 x i64> @llvm.arm.mve.vstr.scatter.base.wb.predicated.v2i64.v2i64.v4i1(<2 x i64>, i32, <2 x i64>, <4 x i1>)
