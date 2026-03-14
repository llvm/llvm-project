; RUN: llc < %s -mtriple=x86_64-- -mcpu=broadwell | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx2 | FileCheck %s

; Check that llc can set function attributes target-cpu and target-features
; using command line options -mcpu and -mattr.

; CHECK: vpsadbw (%r{{si|dx}}), %ymm{{[0-9]+}}, %ymm{{[0-9]+}}

define <4 x i64> @foo1(ptr %s1, ptr %s2) {
entry:
  %ps1 = load <4 x i64>, ptr %s1
  %ps2 = load <4 x i64>, ptr %s2
  %0 = bitcast <4 x i64> %ps1 to <32 x i8>
  %1 = bitcast <4 x i64> %ps2 to <32 x i8>
  %2 = tail call <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8> %0, <32 x i8> %1)
  ret <4 x i64> %2
}

declare <4 x i64> @llvm.x86.avx2.psad.bw(<32 x i8>, <32 x i8>)
