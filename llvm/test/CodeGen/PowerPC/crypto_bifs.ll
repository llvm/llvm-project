; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+crypto < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck %s
; FIXME: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s
; FIXME: The original intent was to add a check-next for the blr after every check.
; However, this currently fails since we don't eliminate stores of the unused
; locals. These stores are sometimes scheduled after the crypto instruction

; Function Attrs: nounwind
define <16 x i8> @test_vpmsumb() #0 {
entry:
  %a = alloca <16 x i8>, align 16
  %b = alloca <16 x i8>, align 16
  store <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, <16 x i8>* %a, align 16
  store <16 x i8> <i8 113, i8 114, i8 115, i8 116, i8 117, i8 118, i8 119, i8 120, i8 121, i8 122, i8 123, i8 124, i8 125, i8 126, i8 127, i8 112>, <16 x i8>* %b, align 16
  %0 = load <16 x i8>,  <16 x i8>* %a, align 16
  %1 = load <16 x i8>,  <16 x i8>* %b, align 16
  %2 = call <16 x i8> @llvm.ppc.altivec.crypto.vpmsumb(<16 x i8> %0, <16 x i8> %1)
  ret <16 x i8> %2
; CHECK: vpmsumb 2,
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.crypto.vpmsumb(<16 x i8>, <16 x i8>) #1

; Function Attrs: nounwind
define <8 x i16> @test_vpmsumh() #0 {
entry:
  %a = alloca <8 x i16>, align 16
  %b = alloca <8 x i16>, align 16
  store <8 x i16> <i16 258, i16 772, i16 1286, i16 1800, i16 2314, i16 2828, i16 3342, i16 3856>, <8 x i16>* %a, align 16
  store <8 x i16> <i16 29042, i16 29556, i16 30070, i16 30584, i16 31098, i16 31612, i16 32126, i16 32624>, <8 x i16>* %b, align 16
  %0 = load <8 x i16>,  <8 x i16>* %a, align 16
  %1 = load <8 x i16>,  <8 x i16>* %b, align 16
  %2 = call <8 x i16> @llvm.ppc.altivec.crypto.vpmsumh(<8 x i16> %0, <8 x i16> %1)
  ret <8 x i16> %2
; CHECK: vpmsumh 2,
}

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.crypto.vpmsumh(<8 x i16>, <8 x i16>) #1

; Function Attrs: nounwind
define <4 x i32> @test_vpmsumw() #0 {
entry:
  %a = alloca <4 x i32>, align 16
  %b = alloca <4 x i32>, align 16
  store <4 x i32> <i32 16909060, i32 84281096, i32 151653132, i32 219025168>, <4 x i32>* %a, align 16
  store <4 x i32> <i32 1903326068, i32 1970698104, i32 2038070140, i32 2105442160>, <4 x i32>* %b, align 16
  %0 = load <4 x i32>,  <4 x i32>* %a, align 16
  %1 = load <4 x i32>,  <4 x i32>* %b, align 16
  %2 = call <4 x i32> @llvm.ppc.altivec.crypto.vpmsumw(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %2
; CHECK: vpmsumw 2,
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.crypto.vpmsumw(<4 x i32>, <4 x i32>) #1

; Function Attrs: nounwind
define <2 x i64> @test_vpmsumd() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = load <2 x i64>,  <2 x i64>* %b, align 16
  %2 = call <2 x i64> @llvm.ppc.altivec.crypto.vpmsumd(<2 x i64> %0, <2 x i64> %1)
  ret <2 x i64> %2
; CHECK: vpmsumd 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vpmsumd(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind
define <2 x i64> @test_vsbox() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = call <2 x i64> @llvm.ppc.altivec.crypto.vsbox(<2 x i64> %0)
  ret <2 x i64> %1
; CHECK: vsbox 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vsbox(<2 x i64>) #1

; Function Attrs: nounwind
define <16 x i8> @test_vpermxorb() #0 {
entry:
  %a = alloca <16 x i8>, align 16
  %b = alloca <16 x i8>, align 16
  %c = alloca <16 x i8>, align 16
  store <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, <16 x i8>* %a, align 16
  store <16 x i8> <i8 113, i8 114, i8 115, i8 116, i8 117, i8 118, i8 119, i8 120, i8 121, i8 122, i8 123, i8 124, i8 125, i8 126, i8 127, i8 112>, <16 x i8>* %b, align 16
  store <16 x i8> <i8 113, i8 114, i8 115, i8 116, i8 117, i8 118, i8 119, i8 120, i8 121, i8 122, i8 123, i8 124, i8 125, i8 126, i8 127, i8 112>, <16 x i8>* %c, align 16
  %0 = load <16 x i8>,  <16 x i8>* %a, align 16
  %1 = load <16 x i8>,  <16 x i8>* %b, align 16
  %2 = load <16 x i8>,  <16 x i8>* %c, align 16
  %3 = call <16 x i8> @llvm.ppc.altivec.crypto.vpermxor(<16 x i8> %0, <16 x i8> %1, <16 x i8> %2)
  ret <16 x i8> %3
; CHECK: vpermxor 2,
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.crypto.vpermxor(<16 x i8>, <16 x i8>, <16 x i8>) #1

; Function Attrs: nounwind
define <8 x i16> @test_vpermxorh() #0 {
entry:
  %a = alloca <8 x i16>, align 16
  %b = alloca <8 x i16>, align 16
  %c = alloca <8 x i16>, align 16
  store <8 x i16> <i16 258, i16 772, i16 1286, i16 1800, i16 2314, i16 2828, i16 3342, i16 3856>, <8 x i16>* %a, align 16
  store <8 x i16> <i16 29042, i16 29556, i16 30070, i16 30584, i16 31098, i16 31612, i16 32126, i16 32624>, <8 x i16>* %b, align 16
  store <8 x i16> <i16 29042, i16 29556, i16 30070, i16 30584, i16 31098, i16 31612, i16 32126, i16 32624>, <8 x i16>* %c, align 16
  %0 = load <8 x i16>,  <8 x i16>* %a, align 16
  %1 = bitcast <8 x i16> %0 to <16 x i8>
  %2 = load <8 x i16>,  <8 x i16>* %b, align 16
  %3 = bitcast <8 x i16> %2 to <16 x i8>
  %4 = load <8 x i16>,  <8 x i16>* %c, align 16
  %5 = bitcast <8 x i16> %4 to <16 x i8>
  %6 = call <16 x i8> @llvm.ppc.altivec.crypto.vpermxor(<16 x i8> %1, <16 x i8> %3, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <8 x i16>
  ret <8 x i16> %7
; CHECK: vpermxor 2,
}

; Function Attrs: nounwind
define <4 x i32> @test_vpermxorw() #0 {
entry:
  %a = alloca <4 x i32>, align 16
  %b = alloca <4 x i32>, align 16
  %c = alloca <4 x i32>, align 16
  store <4 x i32> <i32 16909060, i32 84281096, i32 151653132, i32 219025168>, <4 x i32>* %a, align 16
  store <4 x i32> <i32 1903326068, i32 1970698104, i32 2038070140, i32 2105442160>, <4 x i32>* %b, align 16
  store <4 x i32> <i32 1903326068, i32 1970698104, i32 2038070140, i32 2105442160>, <4 x i32>* %c, align 16
  %0 = load <4 x i32>,  <4 x i32>* %a, align 16
  %1 = bitcast <4 x i32> %0 to <16 x i8>
  %2 = load <4 x i32>,  <4 x i32>* %b, align 16
  %3 = bitcast <4 x i32> %2 to <16 x i8>
  %4 = load <4 x i32>,  <4 x i32>* %c, align 16
  %5 = bitcast <4 x i32> %4 to <16 x i8>
  %6 = call <16 x i8> @llvm.ppc.altivec.crypto.vpermxor(<16 x i8> %1, <16 x i8> %3, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <4 x i32>
  ret <4 x i32> %7
; CHECK: vpermxor 2,
}

; Function Attrs: nounwind
define <2 x i64> @test_vpermxord() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  %c = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %c, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = bitcast <2 x i64> %0 to <16 x i8>
  %2 = load <2 x i64>,  <2 x i64>* %b, align 16
  %3 = bitcast <2 x i64> %2 to <16 x i8>
  %4 = load <2 x i64>,  <2 x i64>* %c, align 16
  %5 = bitcast <2 x i64> %4 to <16 x i8>
  %6 = call <16 x i8> @llvm.ppc.altivec.crypto.vpermxor(<16 x i8> %1, <16 x i8> %3, <16 x i8> %5)
  %7 = bitcast <16 x i8> %6 to <2 x i64>
  ret <2 x i64> %7
; CHECK: vpermxor 2,
}

; Function Attrs: nounwind
define <2 x i64> @test_vcipher() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = load <2 x i64>,  <2 x i64>* %b, align 16
  %2 = call <2 x i64> @llvm.ppc.altivec.crypto.vcipher(<2 x i64> %0, <2 x i64> %1)
  ret <2 x i64> %2
; CHECK: vcipher 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vcipher(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind
define <2 x i64> @test_vcipherlast() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = load <2 x i64>,  <2 x i64>* %b, align 16
  %2 = call <2 x i64> @llvm.ppc.altivec.crypto.vcipherlast(<2 x i64> %0, <2 x i64> %1)
  ret <2 x i64> %2
; CHECK: vcipherlast 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vcipherlast(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind
define <2 x i64> @test_vncipher() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = load <2 x i64>,  <2 x i64>* %b, align 16
  %2 = call <2 x i64> @llvm.ppc.altivec.crypto.vncipher(<2 x i64> %0, <2 x i64> %1)
  ret <2 x i64> %2
; CHECK: vncipher 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vncipher(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind
define <2 x i64> @test_vncipherlast() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  %b = alloca <2 x i64>, align 16
  store <2 x i64> <i64 72623859790382856, i64 651345242494996240>, <2 x i64>* %a, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %b, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = load <2 x i64>,  <2 x i64>* %b, align 16
  %2 = call <2 x i64> @llvm.ppc.altivec.crypto.vncipherlast(<2 x i64> %0, <2 x i64> %1)
  ret <2 x i64> %2
; CHECK: vncipherlast 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vncipherlast(<2 x i64>, <2 x i64>) #1

; Function Attrs: nounwind
define <4 x i32> @test_vshasigmaw() #0 {
entry:
  %a = alloca <4 x i32>, align 16
  store <4 x i32> <i32 16909060, i32 84281096, i32 151653132, i32 219025168>, <4 x i32>* %a, align 16
  %0 = load <4 x i32>,  <4 x i32>* %a, align 16
  %1 = call <4 x i32> @llvm.ppc.altivec.crypto.vshasigmaw(<4 x i32> %0, i32 1, i32 15)
  ret <4 x i32> %1
; CHECK: vshasigmaw 2,
}

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.crypto.vshasigmaw(<4 x i32>, i32, i32) #1

; Function Attrs: nounwind
define <2 x i64> @test_vshasigmad() #0 {
entry:
  %a = alloca <2 x i64>, align 16
  store <2 x i64> <i64 8174723217654970232, i64 8753444600359583600>, <2 x i64>* %a, align 16
  %0 = load <2 x i64>,  <2 x i64>* %a, align 16
  %1 = call <2 x i64> @llvm.ppc.altivec.crypto.vshasigmad(<2 x i64> %0, i32 1, i32 15)
  ret <2 x i64> %1
; CHECK: vshasigmad 2,
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.crypto.vshasigmad(<2 x i64>, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 230949) (llvm/trunk 230946)"}
