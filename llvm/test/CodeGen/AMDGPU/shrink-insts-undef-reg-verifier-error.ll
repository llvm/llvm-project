; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s

define void @bitset_verifier_error() local_unnamed_addr #0 {
bb:
  %i = call float @llvm.fabs.f32(float undef) #0
  %i1 = bitcast float %i to i32
  br label %bb2

bb2:                                              ; preds = %bb
  %i3 = call float @llvm.fabs.f32(float undef) #0
  %i4 = fcmp fast ult float %i3, 0x3FEFF7CEE0000000
  br i1 %i4, label %bb5, label %bb6

bb5:                                              ; preds = %bb2
  unreachable

bb6:                                              ; preds = %bb2
  unreachable
}


declare float @llvm.fabs.f32(float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable willreturn }
