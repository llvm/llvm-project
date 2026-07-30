; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi -mcpu=cortex-a9 -mattr=+neon,+neonfp | FileCheck %s
; PR14824. The test is presented by Jiangning Liu. If the ld/st optimization algorithm is changed, this test case may fail.
; Also if the machine code for ld/st optimizor is changed, this test case may fail. If so, remove this test.

define void @sample_test(ptr %secondSource, ptr %source, ptr %dest) nounwind {
; CHECK: sample_test
; CHECK-NOT: vldmia
; CHECK: add
entry:

; Load %source
  %s0 = load <8 x i64> , ptr %source, align 64
  %arrayidx64 = getelementptr inbounds <8 x i64>, ptr %source, i32 6
  %s120 = load <8 x i64> , ptr %arrayidx64, align 64
  %s122 = bitcast <8 x i64> %s120 to i512
  %data.i.i677.48.extract.shift = lshr i512 %s122, 384
  %data.i.i677.48.extract.trunc = trunc i512 %data.i.i677.48.extract.shift to i64
  %s123 = insertelement <8 x i64> undef, i64 %data.i.i677.48.extract.trunc, i32 0
  %data.i.i677.32.extract.shift = lshr i512 %s122, 256
  %data.i.i677.32.extract.trunc = trunc i512 %data.i.i677.32.extract.shift to i64
  %s124 = insertelement <8 x i64> %s123, i64 %data.i.i677.32.extract.trunc, i32 1
  %data.i.i677.16.extract.shift = lshr i512 %s122, 128
  %data.i.i677.16.extract.trunc = trunc i512 %data.i.i677.16.extract.shift to i64
  %s125 = insertelement <8 x i64> %s124, i64 %data.i.i677.16.extract.trunc, i32 2
  %data.i.i677.56.extract.shift = lshr i512 %s122, 448
  %data.i.i677.56.extract.trunc = trunc i512 %data.i.i677.56.extract.shift to i64
  %s126 = insertelement <8 x i64> %s125, i64 %data.i.i677.56.extract.trunc, i32 3
  %data.i.i677.24.extract.shift = lshr i512 %s122, 192
  %data.i.i677.24.extract.trunc = trunc i512 %data.i.i677.24.extract.shift to i64
  %s127 = insertelement <8 x i64> %s126, i64 %data.i.i677.24.extract.trunc, i32 4
  %s128 = insertelement <8 x i64> %s127, i64 %data.i.i677.32.extract.trunc, i32 5
  %s129 = insertelement <8 x i64> %s128, i64 %data.i.i677.16.extract.trunc, i32 6
  %s130 = insertelement <8 x i64> %s129, i64 %data.i.i677.56.extract.trunc, i32 7

; Load %secondSource
  %s1 = load <8 x i64> , ptr %secondSource, align 64
  %arrayidx67 = getelementptr inbounds <8 x i64>, ptr %secondSource, i32 6
  %s121 = load <8 x i64> , ptr %arrayidx67, align 64
  %s131 = bitcast <8 x i64> %s121 to i512
  %data.i1.i676.48.extract.shift = lshr i512 %s131, 384
  %data.i1.i676.48.extract.trunc = trunc i512 %data.i1.i676.48.extract.shift to i64
  %s132 = insertelement <8 x i64> undef, i64 %data.i1.i676.48.extract.trunc, i32 0
  %data.i1.i676.32.extract.shift = lshr i512 %s131, 256
  %data.i1.i676.32.extract.trunc = trunc i512 %data.i1.i676.32.extract.shift to i64
  %s133 = insertelement <8 x i64> %s132, i64 %data.i1.i676.32.extract.trunc, i32 1
  %data.i1.i676.16.extract.shift = lshr i512 %s131, 128
  %data.i1.i676.16.extract.trunc = trunc i512 %data.i1.i676.16.extract.shift to i64
  %s134 = insertelement <8 x i64> %s133, i64 %data.i1.i676.16.extract.trunc, i32 2
  %data.i1.i676.56.extract.shift = lshr i512 %s131, 448
  %data.i1.i676.56.extract.trunc = trunc i512 %data.i1.i676.56.extract.shift to i64
  %s135 = insertelement <8 x i64> %s134, i64 %data.i1.i676.56.extract.trunc, i32 3
  %data.i1.i676.24.extract.shift = lshr i512 %s131, 192
  %data.i1.i676.24.extract.trunc = trunc i512 %data.i1.i676.24.extract.shift to i64
  %s136 = insertelement <8 x i64> %s135, i64 %data.i1.i676.24.extract.trunc, i32 4
  %s137 = insertelement <8 x i64> %s136, i64 %data.i1.i676.32.extract.trunc, i32 5
  %s138 = insertelement <8 x i64> %s137, i64 %data.i1.i676.16.extract.trunc, i32 6
  %s139 = insertelement <8 x i64> %s138, i64 %data.i1.i676.56.extract.trunc, i32 7

; Operations about %Source and %secondSource
  %vecinit28.i.i699 = shufflevector <8 x i64> %s139, <8 x i64> %s130, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 undef, i32 undef, i32 undef>
  %vecinit35.i.i700 = shufflevector <8 x i64> %vecinit28.i.i699, <8 x i64> %s139, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 13, i32 undef, i32 undef>
  %vecinit42.i.i701 = shufflevector <8 x i64> %vecinit35.i.i700, <8 x i64> %s139, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 14, i32 undef>
  %vecinit49.i.i702 = shufflevector <8 x i64> %vecinit42.i.i701, <8 x i64> %s130, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 15>
  %arrayidx72 = getelementptr inbounds <8 x i64>, ptr %dest, i32 6
  store <8 x i64> %vecinit49.i.i702, ptr %arrayidx72, align 64
  %arrayidx78 = getelementptr inbounds <8 x i64>, ptr %secondSource, i32 7
  %s141 = load <8 x i64> , ptr %arrayidx78, align 64
  %s151 = bitcast <8 x i64> %s141 to i512
  %data.i1.i649.32.extract.shift = lshr i512 %s151, 256
  %data.i1.i649.32.extract.trunc = trunc i512 %data.i1.i649.32.extract.shift to i64
  %s152 = insertelement <8 x i64> undef, i64 %data.i1.i649.32.extract.trunc, i32 0
  %s153 = insertelement <8 x i64> %s152, i64 %data.i1.i649.32.extract.trunc, i32 1
  %data.i1.i649.16.extract.shift = lshr i512 %s151, 128
  %data.i1.i649.16.extract.trunc = trunc i512 %data.i1.i649.16.extract.shift to i64
  %s154 = insertelement <8 x i64> %s153, i64 %data.i1.i649.16.extract.trunc, i32 2
  %data.i1.i649.8.extract.shift = lshr i512 %s151, 64
  %data.i1.i649.8.extract.trunc = trunc i512 %data.i1.i649.8.extract.shift to i64
  %s155 = insertelement <8 x i64> %s154, i64 %data.i1.i649.8.extract.trunc, i32 3
  %arrayidx83 = getelementptr inbounds <8 x i64>, ptr %dest, i32 7
  store <8 x i64> %s155, ptr %arrayidx83, align 64
  ret void
}
