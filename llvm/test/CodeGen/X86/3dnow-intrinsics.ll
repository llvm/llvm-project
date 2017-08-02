; RUN: llc < %s -mtriple=i686-- -mattr=+3dnow | FileCheck %s

define <8 x i8> @test_pavgusb(x86_mmx %a.coerce, x86_mmx %b.coerce) nounwind readnone {
; CHECK: pavgusb
entry:
  %0 = bitcast x86_mmx %a.coerce to <8 x i8>
  %1 = bitcast x86_mmx %b.coerce to <8 x i8>
  %2 = bitcast <8 x i8> %0 to x86_mmx
  %3 = bitcast <8 x i8> %1 to x86_mmx
  %4 = call x86_mmx @llvm.x86.3dnow.pavgusb(x86_mmx %2, x86_mmx %3)
  %5 = bitcast x86_mmx %4 to <8 x i8>
  ret <8 x i8> %5
}

declare x86_mmx @llvm.x86.3dnow.pavgusb(x86_mmx, x86_mmx) nounwind readnone

define <2 x i32> @test_pf2id(<2 x float> %a) nounwind readnone {
; CHECK: pf2id
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnow.pf2id(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x i32>
  ret <2 x i32> %2
}

declare x86_mmx @llvm.x86.3dnow.pf2id(x86_mmx) nounwind readnone

define <2 x float> @test_pfacc(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfacc
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfacc(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfacc(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfadd(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfadd
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfadd(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfadd(x86_mmx, x86_mmx) nounwind readnone

define <2 x i32> @test_pfcmpeq(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfcmpeq
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfcmpeq(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x i32>
  ret <2 x i32> %3
}

declare x86_mmx @llvm.x86.3dnow.pfcmpeq(x86_mmx, x86_mmx) nounwind readnone

define <2 x i32> @test_pfcmpge(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfcmpge
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfcmpge(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x i32>
  ret <2 x i32> %3
}

declare x86_mmx @llvm.x86.3dnow.pfcmpge(x86_mmx, x86_mmx) nounwind readnone

define <2 x i32> @test_pfcmpgt(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfcmpgt
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfcmpgt(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x i32>
  ret <2 x i32> %3
}

declare x86_mmx @llvm.x86.3dnow.pfcmpgt(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfmax(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfmax
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfmax(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfmax(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfmin(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfmin
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfmin(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfmin(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfmul(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfmul
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfmul(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfmul(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfrcp(<2 x float> %a) nounwind readnone {
; CHECK: pfrcp
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnow.pfrcp(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x float>
  ret <2 x float> %2
}

declare x86_mmx @llvm.x86.3dnow.pfrcp(x86_mmx) nounwind readnone

define <2 x float> @test_pfrcpit1(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfrcpit1
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfrcpit1(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfrcpit1(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfrcpit2(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfrcpit2
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfrcpit2(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfrcpit2(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfrsqrt(<2 x float> %a) nounwind readnone {
; CHECK: pfrsqrt
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnow.pfrsqrt(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x float>
  ret <2 x float> %2
}

declare x86_mmx @llvm.x86.3dnow.pfrsqrt(x86_mmx) nounwind readnone

define <2 x float> @test_pfrsqit1(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfrsqit1
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfrsqit1(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfrsqit1(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfsub(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfsub
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfsub(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfsub(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfsubr(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfsubr
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnow.pfsubr(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pfsubr(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pi2fd(x86_mmx %a.coerce) nounwind readnone {
; CHECK: pi2fd
entry:
  %0 = bitcast x86_mmx %a.coerce to <2 x i32>
  %1 = bitcast <2 x i32> %0 to x86_mmx
  %2 = call x86_mmx @llvm.x86.3dnow.pi2fd(x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnow.pi2fd(x86_mmx) nounwind readnone

define <4 x i16> @test_pmulhrw(x86_mmx %a.coerce, x86_mmx %b.coerce) nounwind readnone {
; CHECK: pmulhrw
entry:
  %0 = bitcast x86_mmx %a.coerce to <4 x i16>
  %1 = bitcast x86_mmx %b.coerce to <4 x i16>
  %2 = bitcast <4 x i16> %0 to x86_mmx
  %3 = bitcast <4 x i16> %1 to x86_mmx
  %4 = call x86_mmx @llvm.x86.3dnow.pmulhrw(x86_mmx %2, x86_mmx %3)
  %5 = bitcast x86_mmx %4 to <4 x i16>
  ret <4 x i16> %5
}

declare x86_mmx @llvm.x86.3dnow.pmulhrw(x86_mmx, x86_mmx) nounwind readnone

define <2 x i32> @test_pf2iw(<2 x float> %a) nounwind readnone {
; CHECK: pf2iw
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnowa.pf2iw(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x i32>
  ret <2 x i32> %2
}

declare x86_mmx @llvm.x86.3dnowa.pf2iw(x86_mmx) nounwind readnone

define <2 x float> @test_pfnacc(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfnacc
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnowa.pfnacc(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnowa.pfnacc(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pfpnacc(<2 x float> %a, <2 x float> %b) nounwind readnone {
; CHECK: pfpnacc
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = bitcast <2 x float> %b to x86_mmx
  %2 = tail call x86_mmx @llvm.x86.3dnowa.pfpnacc(x86_mmx %0, x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnowa.pfpnacc(x86_mmx, x86_mmx) nounwind readnone

define <2 x float> @test_pi2fw(x86_mmx %a.coerce) nounwind readnone {
; CHECK: pi2fw
entry:
  %0 = bitcast x86_mmx %a.coerce to <2 x i32>
  %1 = bitcast <2 x i32> %0 to x86_mmx
  %2 = call x86_mmx @llvm.x86.3dnowa.pi2fw(x86_mmx %1)
  %3 = bitcast x86_mmx %2 to <2 x float>
  ret <2 x float> %3
}

declare x86_mmx @llvm.x86.3dnowa.pi2fw(x86_mmx) nounwind readnone

define <2 x float> @test_pswapdsf(<2 x float> %a) nounwind readnone {
; CHECK: pswapd {{.*#+}} mm0 = mem[1,0]
entry:
  %0 = bitcast <2 x float> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnowa.pswapd(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x float>
  ret <2 x float> %2
}

define <2 x i32> @test_pswapdsi(<2 x i32> %a) nounwind readnone {
; CHECK: pswapd {{.*#+}} mm0 = mem[1,0]
entry:
  %0 = bitcast <2 x i32> %a to x86_mmx
  %1 = tail call x86_mmx @llvm.x86.3dnowa.pswapd(x86_mmx %0)
  %2 = bitcast x86_mmx %1 to <2 x i32>
  ret <2 x i32> %2
}

declare x86_mmx @llvm.x86.3dnowa.pswapd(x86_mmx) nounwind readnone
