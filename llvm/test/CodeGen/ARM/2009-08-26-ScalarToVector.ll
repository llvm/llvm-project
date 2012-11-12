; RUN: llc < %s -mattr=+neon | not grep fldmfdd
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

%bar = type { float, float, float }
%baz = type { i32, [16 x %bar], [16 x float], [16 x i32], i8 }
%foo = type { <4 x float> }
%quux = type { i32 (...)**, %baz*, i32 }
%quuz = type { %quux, i32, %bar, [128 x i8], [16 x %foo], %foo, %foo, %foo }

declare <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define void @_ZN6squish10ClusterFit9Compress3EPv(%quuz* %this, i8* %block) {
entry:
  %0 = lshr <4 x i32> zeroinitializer, <i32 31, i32 31, i32 31, i32 31> ; <<4 x i32>> [#uses=1]
  %1 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 2, i32 3> ; <<2 x i32>> [#uses=1]
  %2 = call <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32> undef, <2 x i32> %1) nounwind ; <<2 x i32>> [#uses=1]
  %3 = extractelement <2 x i32> %2, i32 0         ; <i32> [#uses=1]
  %not..i = icmp eq i32 %3, undef                 ; <i1> [#uses=1]
  br i1 %not..i, label %return, label %bb221

bb221:                                            ; preds = %bb221, %entry
  br label %bb221

return:                                           ; preds = %entry
  ret void
}
