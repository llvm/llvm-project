; RUN: llc -mtriple thumbv7---elf -mattr=+neon -filetype asm -o - %s | FileCheck %s

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
  %0 = lshr <4 x i32> zeroinitializer, <i32 31, i32 31, i32 31, i32 31>
  %1 = shufflevector <4 x i32> %0, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %2 = call <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32> undef, <2 x i32> %1) nounwind
  %3 = extractelement <2 x i32> %2, i32 0
  %not..i = icmp eq i32 %3, undef
  br i1 %not..i, label %return, label %bb221

bb221:
  br label %bb221

return:
  ret void
}

; CHECK-NOT: fldmfdd

