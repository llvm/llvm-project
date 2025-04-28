; RUN: llc -mtriple=amdgcn < %s | FileCheck %s
; REQUIRES: asserts
;
; This test used to crash with the following assertion:
; llc: include/llvm/ADT/IntervalMap.h:632: unsigned int llvm::IntervalMapImpl::LeafNode<llvm::SlotIndex, llvm::LiveInterval *, 8, llvm::IntervalMapInfo<llvm::SlotIndex> >::insertFrom(unsigned int &, unsigned int, KeyT, KeyT, ValT) [KeyT = llvm::SlotIndex, ValT = llvm::LiveInterval *, N = 8, Traits = llvm::IntervalMapInfo<llvm::SlotIndex>]: Assertion `(i == Size || Traits::stopLess(b, start(i))) && "Overlapping insert"' failed.
;
; This was related to incorrectly calculating subregister live ranges
; (i.e. live interval subranges): subregister defs are not uses for that
; purpose.
;
; Check for a valid output:
; CHECK: tbuffer_store_format_x

target triple = "amdgcn--"

define amdgpu_gs void @main(i32 inreg %arg) #0 {
main_body:
  %tmp = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 20, i32 0)
  %tmp1 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 24, i32 0)
  %tmp2 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 48, i32 0)
  %array_vector3 = insertelement <4 x float> zeroinitializer, float %tmp2, i32 3
  %array_vector5 = insertelement <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, float %tmp, i32 1
  %array_vector6 = insertelement <4 x float> %array_vector5, float undef, i32 2
  %array_vector9 = insertelement <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, float %tmp1, i32 1
  %array_vector10 = insertelement <4 x float> %array_vector9, float 0.000000e+00, i32 2
  %array_vector11 = insertelement <4 x float> %array_vector10, float undef, i32 3
  %tmp3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> undef, i32 undef, i32 4864, i32 0)
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 %tmp3, <4 x i32> undef, i32 36, i32 %arg, i32 68, i32 3)
  %bc = bitcast <4 x float> %array_vector3 to <4 x i32>
  %tmp4 = extractelement <4 x i32> %bc, i32 undef
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 %tmp4, <4 x i32> undef, i32 48, i32 %arg, i32 68, i32 3)
  %bc49 = bitcast <4 x float> %array_vector11 to <4 x i32>
  %tmp5 = extractelement <4 x i32> %bc49, i32 undef
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 %tmp5, <4 x i32> undef, i32 72, i32 %arg, i32 68, i32 3)
  %array_vector21 = insertelement <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, float %tmp, i32 1
  %array_vector22 = insertelement <4 x float> %array_vector21, float undef, i32 2
  %array_vector23 = insertelement <4 x float> %array_vector22, float undef, i32 3
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 undef, <4 x i32> undef, i32 28, i32 %arg, i32 68, i32 3)
  %bc52 = bitcast <4 x float> %array_vector23 to <4 x i32>
  %tmp6 = extractelement <4 x i32> %bc52, i32 undef
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 %tmp6, <4 x i32> undef, i32 64, i32 %arg, i32 68, i32 3)
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 undef, <4 x i32> undef, i32 20, i32 %arg, i32 68, i32 3)
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 undef, <4 x i32> undef, i32 56, i32 %arg, i32 68, i32 3)
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 undef, <4 x i32> undef, i32 92, i32 %arg, i32 68, i32 3)
  ret void
}

declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32 immarg) #1
declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.raw.tbuffer.store.i32(i32, <4 x i32>, i32, i32, i32 immarg, i32 immarg) #3

attributes #0 = { nounwind "target-cpu"="tonga" }
attributes #1 = { nounwind readnone willreturn }
attributes #2 = { nounwind readonly willreturn }
attributes #3 = { nounwind willreturn writeonly }
