; RUN: llc < %s -mtriple=thumbv7s-apple-ios3.0.0 -mcpu=generic | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc < %s -mtriple=thumbeb -target-abi apcs -mattr=v7,neon | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE

; PR15525
; CHECK-LABEL: test1:
; CHECK: ldr.w	[[REG:r[0-9]+]], [sp]
; CHECK-LE-NEXT: vmov	{{d[0-9]+}}, r1, r2
; CHECK-LE-NEXT: vmov	{{d[0-9]+}}, r3, [[REG]]
; CHECK-BE-NEXT: vmov	{{d[0-9]+}}, r2, r1
; CHECK-BE: vmov	{{d[0-9]+}}, [[REG]], r3
; CHECK: vst1.8	{{{d[0-9]+}}, {{d[0-9]+}}}, [r0]
; CHECK-NEXT: bx	lr
define void @test1(ptr %arg, [4 x i64] %vec.coerce) {
bb:
  %tmp = extractvalue [4 x i64] %vec.coerce, 0
  %tmp2 = bitcast i64 %tmp to <8 x i8>
  %tmp3 = shufflevector <8 x i8> %tmp2, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %tmp4 = extractvalue [4 x i64] %vec.coerce, 1
  %tmp5 = bitcast i64 %tmp4 to <8 x i8>
  %tmp6 = shufflevector <8 x i8> %tmp5, <8 x i8> undef, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp7 = shufflevector <16 x i8> %tmp6, <16 x i8> %tmp3, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.arm.neon.vst1.p0.v16i8(ptr %arg, <16 x i8> %tmp7, i32 2)
  ret void
}

declare void @llvm.arm.neon.vst1.p0.v16i8(ptr, <16 x i8>, i32)
