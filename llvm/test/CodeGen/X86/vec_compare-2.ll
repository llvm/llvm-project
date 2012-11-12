; RUN: llc < %s -mtriple=i686-linux -mcpu=penryn | FileCheck %s

declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

declare <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define void @blackDespeckle_wrapper(i8** %args_list, i64* %gtid, i64 %xend) {
entry:
; CHECK: cfi_def_cfa_offset
; CHECK-NOT: set
; CHECK: pmovzxwq
; CHECK: pshufb
  %shr.i = ashr <4 x i32> zeroinitializer, <i32 3, i32 3, i32 3, i32 3> ; <<4 x i32>> [#uses=1]
  %cmp318.i = sext <4 x i1> zeroinitializer to <4 x i32> ; <<4 x i32>> [#uses=1]
  %sub322.i = sub <4 x i32> %shr.i, zeroinitializer ; <<4 x i32>> [#uses=1]
  %cmp323.x = icmp slt <4 x i32> zeroinitializer, %sub322.i ; <<4 x i1>> [#uses=1]
  %cmp323.i = sext <4 x i1> %cmp323.x to <4 x i32> ; <<4 x i32>> [#uses=1]
  %or.i = or <4 x i32> %cmp318.i, %cmp323.i       ; <<4 x i32>> [#uses=1]
  %tmp10.i83.i = bitcast <4 x i32> %or.i to <4 x float> ; <<4 x float>> [#uses=1]
  %0 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> undef, <4 x float> undef, <4 x float> %tmp10.i83.i) nounwind ; <<4 x float>> [#uses=1]
  %conv.i.i15.i = bitcast <4 x float> %0 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %swz.i.i28.i = shufflevector <4 x i32> %conv.i.i15.i, <4 x i32> undef, <2 x i32> <i32 0, i32 1> ; <<2 x i32>> [#uses=1]
  %tmp6.i29.i = bitcast <2 x i32> %swz.i.i28.i to <4 x i16> ; <<4 x i16>> [#uses=1]
  %swz.i30.i = shufflevector <4 x i16> %tmp6.i29.i, <4 x i16> undef, <2 x i32> <i32 0, i32 1> ; <<2 x i16>> [#uses=1]
  store <2 x i16> %swz.i30.i, <2 x i16>* undef
  unreachable
  ret void
}
