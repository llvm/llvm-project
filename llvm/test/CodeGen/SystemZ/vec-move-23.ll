; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s -check-prefixes=CHECK,Z14
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s -check-prefixes=CHECK,Z15
;
; Check that int-to-fp conversions from a narrower type get a vector extension.

define void @fun0(<2 x i8> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun0:
; CHECK:      vuphb	%v0, %v24
; CHECK-NEXT: vuphh	%v0, %v0
; CHECK-NEXT: vuphf	%v0, %v0
; CHECK-NEXT: vcdgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = sitofp <2 x i8> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun1(<2 x i16> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun1:
; CHECK:      vuphh	%v0, %v24
; CHECK-NEXT: vuphf	%v0, %v0
; CHECK-NEXT: vcdgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = sitofp <2 x i16> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun2(<2 x i32> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun2:
; CHECK:      vuphf	%v0, %v24
; CHECK-NEXT: vcdgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = sitofp <2 x i32> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun3(<4 x i16> %Src, <4 x float>* %Dst) {
; CHECK-LABEL: fun3:

; Z14:      vuphh	%v0, %v24
; Z14-NEXT: vlgvf	%r0, %v0, 3
; Z14-NEXT: cefbr	%f1, %r0
; Z14-NEXT: vlgvf	%r0, %v0, 2
; Z14-NEXT: cefbr	%f2, %r0
; Z14-NEXT: vlgvf	%r0, %v0, 1
; Z14-NEXT: vmrhf	%v1, %v2, %v1
; Z14-NEXT: cefbr	%f2, %r0
; Z14-NEXT: vlgvf	%r0, %v0, 0
; Z14-NEXT: cefbr	%f0, %r0
; Z14-NEXT: vmrhf	%v0, %v0, %v2
; Z14-NEXT: vmrhg	%v0, %v0, %v1
; Z14-NEXT: vst	%v0, 0(%r2), 3
; Z14-NEXT: br	%r14

; Z15:      vuphh	%v0, %v24
; Z15-NEXT: vcefb	%v0, %v0, 0, 0
; Z15-NEXT: vst	        %v0, 0(%r2), 3
; Z15-NEXT: br	        %r14
  %c = sitofp <4 x i16> %Src to <4 x float>
  store <4 x float> %c, <4 x float>* %Dst
  ret void
}

define void @fun4(<2 x i8> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun4:
; CHECK:      larl	%r1, .LCPI4_0
; CHECK-NEXT: vl	%v0, 0(%r1), 3
; CHECK-NEXT: vperm	%v0, %v0, %v24, %v0
; CHECK-NEXT: vcdlgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = uitofp <2 x i8> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun5(<2 x i16> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun5:
; CHECK:      larl	%r1, .LCPI5_0
; CHECK-NEXT: vl	%v0, 0(%r1), 3
; CHECK-NEXT: vperm	%v0, %v0, %v24, %v0
; CHECK-NEXT: vcdlgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = uitofp <2 x i16> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun6(<2 x i32> %Src, <2 x double>* %Dst) {
; CHECK-LABEL: fun6:
; CHECK:      vuplhf	%v0, %v24
; CHECK-NEXT: vcdlgb	%v0, %v0, 0, 0
; CHECK-NEXT: vst	%v0, 0(%r2), 3
; CHECK-NEXT: br	%r14
  %c = uitofp <2 x i32> %Src to <2 x double>
  store <2 x double> %c, <2 x double>* %Dst
  ret void
}

define void @fun7(<4 x i16> %Src, <4 x float>* %Dst) {
; CHECK-LABEL: fun7:

; Z14:      vuplhh	%v0, %v24
; Z14-NEXT: vlgvf	%r0, %v0, 3
; Z14-NEXT: celfbr	%f1, 0, %r0, 0
; Z14-NEXT: vlgvf	%r0, %v0, 2
; Z14-NEXT: celfbr	%f2, 0, %r0, 0
; Z14-NEXT: vlgvf	%r0, %v0, 1
; Z14-NEXT: vmrhf	%v1, %v2, %v1
; Z14-NEXT: celfbr	%f2, 0, %r0, 0
; Z14-NEXT: vlgvf	%r0, %v0, 0
; Z14-NEXT: celfbr	%f0, 0, %r0, 0
; Z14-NEXT: vmrhf	%v0, %v0, %v2
; Z14-NEXT: vmrhg	%v0, %v0, %v1
; Z14-NEXT: vst	%v0, 0(%r2), 3
; Z14-NEXT: br	%r14

; Z15:      vuplhh	%v0, %v24
; Z15-NEXT: vcelfb	%v0, %v0, 0, 0
; Z15-NEXT: vst	        %v0, 0(%r2), 3
; Z15-NEXT: br	        %r14
  %c = uitofp <4 x i16> %Src to <4 x float>
  store <4 x float> %c, <4 x float>* %Dst
  ret void
}

