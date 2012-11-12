; RUN: llc %s -mtriple=thumbv7-apple-darwin -verify-machineinstrs -mcpu=cortex-a9 -O0 -o -
; Make sure that the VMOVQQQQ pseudo instruction is handled properly
; by codegen.

define void @test_vmovqqqq_pseudo() nounwind ssp {
entry:
  %vld3_lane = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8* undef, <8 x i16> undef, <8 x i16> undef, <8 x i16> zeroinitializer, i32 7, i32 2)
  store { <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, { <8 x i16>, <8 x i16>, <8 x i16> }* undef
  ret void
}

declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
