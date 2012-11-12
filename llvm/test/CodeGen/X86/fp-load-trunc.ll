; RUN: llc < %s -march=x86 -mcpu=corei7 | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=core-avx-i | FileCheck %s --check-prefix=AVX

define <1 x float> @test1(<1 x double>* %p) nounwind {
; CHECK: test1
; CHECK: cvtsd2ss
; CHECK: ret
; AVX:   test1
; AVX:   vcvtsd2ss
; AVX:   ret
  %x = load <1 x double>* %p
  %y = fptrunc <1 x double> %x to <1 x float>
  ret <1 x float> %y
}

define <2 x float> @test2(<2 x double>* %p) nounwind {
; CHECK: test2
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: ret
; AVX:   test2
; AVX:   vcvtpd2psx {{[0-9]*}}(%{{.*}})
; AVX:   ret
  %x = load <2 x double>* %p
  %y = fptrunc <2 x double> %x to <2 x float>
  ret <2 x float> %y
}

define <4 x float> @test3(<4 x double>* %p) nounwind {
; CHECK: test3
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: movlhps
; CHECK: ret
; AVX:   test3
; AVX:   vcvtpd2psy {{[0-9]*}}(%{{.*}})
; AVX:   ret
  %x = load <4 x double>* %p
  %y = fptrunc <4 x double> %x to <4 x float>
  ret <4 x float> %y
}

define <8 x float> @test4(<8 x double>* %p) nounwind {
; CHECK: test4
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: movlhps
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: cvtpd2ps {{[0-9]*}}(%{{.*}})
; CHECK: movlhps
; CHECK: ret
; AVX:   test4
; AVX:   vcvtpd2psy {{[0-9]*}}(%{{.*}})
; AVX:   vcvtpd2psy {{[0-9]*}}(%{{.*}})
; AVX:   vinsertf128
; AVX:   ret
  %x = load <8 x double>* %p
  %y = fptrunc <8 x double> %x to <8 x float>
  ret <8 x float> %y
}


