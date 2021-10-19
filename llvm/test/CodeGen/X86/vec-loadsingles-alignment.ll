; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s

@e = dso_local global [8 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8], align 16
@d = dso_local global [8 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 16

; The global 'e' has 16 byte alignment, so make sure we don't generate an
; aligned 32-byte load instruction when we combine the load+insert sequence.

define dso_local i32 @subb() nounwind ssp {
; CHECK-LABEL: subb:
; CHECK:  vmovups e(%rip), %ymm
entry:
  %0 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 7), align 4
  %1 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 6), align 8
  %2 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 5), align 4
  %3 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 4), align 16
  %4 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 3), align 4
  %5 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 2), align 8
  %6 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 1), align 4
  %7 = load i32, i32* getelementptr inbounds ([8 x i32], [8 x i32]* @e, i64 0, i64 0), align 16
  %vecinit.i = insertelement <8 x i32> undef, i32 %7, i32 0
  %vecinit1.i = insertelement <8 x i32> %vecinit.i, i32 %6, i32 1
  %vecinit2.i = insertelement <8 x i32> %vecinit1.i, i32 %5, i32 2
  %vecinit3.i = insertelement <8 x i32> %vecinit2.i, i32 %4, i32 3
  %vecinit4.i = insertelement <8 x i32> %vecinit3.i, i32 %3, i32 4
  %vecinit5.i = insertelement <8 x i32> %vecinit4.i, i32 %2, i32 5
  %vecinit6.i = insertelement <8 x i32> %vecinit5.i, i32 %1, i32 6
  %vecinit7.i = insertelement <8 x i32> %vecinit6.i, i32 %0, i32 7
  %8 = bitcast <8 x i32> %vecinit7.i to <32 x i8>
  tail call void @llvm.x86.avx.storeu.dq.256(i8* bitcast ([8 x i32]* @d to i8*), <32 x i8> %8)
  ret i32 0
}

declare void @llvm.x86.avx.storeu.dq.256(i8*, <32 x i8>) nounwind

